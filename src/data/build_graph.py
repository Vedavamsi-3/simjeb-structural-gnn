"""Turn one SimJEB model into a cached surface graph.

Three files describe a bracket, and none of them is sufficient alone:

``<id>.vtk``
    Geometry only -- node coordinates and tetrahedron connectivity. No results.
``<id>field.csv``
    One row per node: coordinates, a ``surf`` flag, and displacement + von Mises for
    all four load cases. The supervision signal.
``<id>.fem``
    The solver deck: where the part is clamped, where it is loaded, and with what.

This module joins them into a single cached graph. The expensive work -- reading a
50 MB mesh, extracting boundary faces, computing normals -- happens once here so that
training epochs only load tensors.

What gets cached, and why
-------------------------
Geometry is stored **once per model** with the targets for **all four load cases**
alongside. Writing one file per (model, load case) would store the same connectivity
four times over; the four cases share identical geometry. Keeping the three unused
cases costs a few MB per model and means changing the training load case never
requires re-running preprocessing.

Edges are stored **undirected** and made bidirectional at load time, and indices are
``int32``. Both are size decisions: the surface of these brackets is ~40% of all nodes
(they are thin and perforated), which is far denser than a solid part, so edge storage
dominates the file.

``edge_attr`` is **not** stored. It is fully derivable from ``pos`` and ``edge_index``,
so storing it would nearly double the file for no information.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from src.data.parse_fem import FemDeck, landmark_centroids, parse_fem

# SimJEB's four load cases, in subcase order. The CSV column prefixes.
LOAD_CASES = ("ver", "hor", "dia", "tor")

# Target channels per load case, in the order they are stored in ``y``.
TARGET_CHANNELS = ("xdisp", "ydisp", "zdisp", "stress")

# Node type codes used in the cached graph.
NODE_NORMAL, NODE_FIXED, NODE_LOADED = 0, 1, 2

# Values of the ``surf`` column in ``<id>field.csv``. It is a four-way classification,
# not a boolean: SimJEB labels the two standardised interfaces separately from plain
# surface. Reading it as 0/1 quietly discards every bolt and lug node.
SURF_INTERIOR, SURF_SURFACE, SURF_BOLT, SURF_LUG = 0, 1, 2, 3


@dataclass
class GraphBuildError(Exception):
    """Raised when a model cannot be turned into a usable graph."""

    model_id: int
    reason: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"model {self.model_id}: {self.reason}"


def boundary_faces(tets: np.ndarray) -> np.ndarray:
    """Triangles on the outer surface: those belonging to exactly one tetrahedron.

    An interior face is shared by two tets; a boundary face by one. Counting how often
    each sorted vertex triple appears separates them.

    Returned with the original (unsorted) winding so that face orientation is
    preserved for normal computation.
    """
    faces = np.concatenate(
        [tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]]
    )
    key = np.sort(faces, axis=1)

    # Structured view so np.unique compares whole rows in one pass -- much faster than
    # unique(axis=0) on a few million rows.
    view = np.ascontiguousarray(key).view(
        np.dtype((np.void, key.dtype.itemsize * key.shape[1]))
    ).ravel()
    _, idx, counts = np.unique(view, return_index=True, return_counts=True)
    return faces[idx[counts == 1]]


def surface_edges(tets: np.ndarray, is_surface: np.ndarray) -> np.ndarray:
    """Undirected edges between surface nodes, taken from tetrahedron edges.

    Returns an ``(E, 2)`` array of global node indices, sorted and deduplicated.

    Why tet edges rather than the boundary triangles' edges
    ------------------------------------------------------
    Every boundary-triangle edge is also a tet edge, so this is a superset. The extra
    edges are the ones that cut *through* material -- across a thin rib, for instance,
    joining a node on one face to the node opposite it. Those connections are real:
    material genuinely joins them, and load genuinely passes between them. Restricting
    to the boundary manifold would force message passing to travel all the way around
    a rib to relate two nodes a millimetre apart.

    What it still refuses to do is connect nodes that merely happen to be close. Two
    faces either side of a gap share no tetrahedron, so no edge appears -- which is
    exactly why mesh connectivity is used here instead of a k-nearest-neighbour graph.
    """
    pairs = np.concatenate(
        [
            tets[:, [0, 1]], tets[:, [0, 2]], tets[:, [0, 3]],
            tets[:, [1, 2]], tets[:, [1, 3]], tets[:, [2, 3]],
        ]
    )
    pairs = pairs[is_surface[pairs[:, 0]] & is_surface[pairs[:, 1]]]
    pairs = np.sort(pairs, axis=1)

    view = np.ascontiguousarray(pairs).view(
        np.dtype((np.void, pairs.dtype.itemsize * 2))
    ).ravel()
    _, idx = np.unique(view, return_index=True)
    return pairs[idx]


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Outward unit normal per vertex, area-weighted from the incident faces.

    A face's orientation relative to the load direction decides whether it sees
    tension, compression or shear, so the normal is genuinely informative -- not
    decoration.

    Falls back to zeros for isolated vertices rather than emitting NaN, which would
    poison the normalisation statistics for every other model in the batch.
    """
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    # Cross product magnitude is twice the triangle area, so summing un-normalised
    # face normals weights each contribution by area for free.
    face_normals = np.cross(v1 - v0, v2 - v0)

    normals = np.zeros_like(vertices)
    for col in range(3):
        np.add.at(normals, faces[:, col], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)


def build_graph(
    model_id: int,
    vtk_path: str | Path,
    csv_path: str | Path,
    fem_path: str | Path,
    spc_set: int = 1,
    load_set: int = 2,
    deck: FemDeck | None = None,
) -> dict:
    """Build the cached surface graph for one model.

    ``load_set`` selects which load case supplies the interface landmarks and the
    fixed/loaded node masks. All four cases share the same clamps and the same load
    lug, so the choice does not affect the features -- targets for every case are
    stored regardless.

    Returns a dict of tensors ready for :func:`torch.save`, plus a ``meta`` dict of
    scalars kept for the QA stage.
    """
    vtk_path, csv_path, fem_path = Path(vtk_path), Path(csv_path), Path(fem_path)

    # ---- read the three sources ------------------------------------------------
    mesh = meshio.read(vtk_path)
    if "tetra" not in mesh.cells_dict:
        raise GraphBuildError(model_id, f"no tetrahedra in {vtk_path.name}")
    tets = mesh.cells_dict["tetra"].astype(np.int32)
    points = np.asarray(mesh.points, dtype=np.float64)

    df = pd.read_csv(csv_path)
    if deck is None:
        deck = parse_fem(fem_path)

    # ---- the alignment assertion ----------------------------------------------
    # Every target attaches to a node by row index. An off-by-one here would attach
    # every label to the wrong node, and nothing would crash: the model would train
    # to a plausible loss curve on pure noise.
    if len(df) != len(points):
        raise GraphBuildError(
            model_id, f"csv has {len(df)} rows but mesh has {len(points)} points"
        )
    coord_error = np.abs(df[["x", "y", "z"]].to_numpy() - points).max()
    if coord_error > 1e-3:
        raise GraphBuildError(
            model_id, f"csv and mesh coordinates disagree by up to {coord_error:.3g} mm"
        )
    deck.validate(n_mesh_nodes=len(points))

    # ---- boundary conditions, as node masks ------------------------------------
    # The deck names abstract reference nodes; these are already expanded out to the
    # real bolt-hole and load-lug mesh nodes by the parser.
    fixed_ids = deck.fixed_mesh_nodes(spc_set)
    loaded_ids = deck.loaded_mesh_nodes(load_set)

    # ``surf`` is a four-way node label, not a boolean -- a detail worth getting right,
    # since reading it as 0/1 silently drops every interface node:
    #     0  interior      1  plain surface
    #     2  bolt-hole interface        3  load-lug interface
    surf_code = df["surf"].to_numpy()
    is_surface = surf_code > 0
    if not is_surface.any():
        raise GraphBuildError(model_id, "no nodes flagged as surface")

    # Cross-check the deck against the dataset's own labels. These are wholly
    # independent sources -- one is the solver's rigid-element definition, the other is
    # SimJEB's node classification -- so agreement is real evidence that the RBE2/RBE3
    # expansion is correct, and disagreement means one of them cannot be trusted for
    # this model.
    for name, ids, code in (
        ("fixed", fixed_ids, SURF_BOLT),
        ("loaded", loaded_ids, SURF_LUG),
    ):
        labelled = np.flatnonzero(surf_code == code) + 1   # to 1-based node ids
        if not np.array_equal(np.sort(ids), labelled):
            raise GraphBuildError(
                model_id,
                f"deck and csv disagree on the {name} interface: deck has "
                f"{len(ids)} nodes, csv labels {len(labelled)}"
            )

    # ---- canonicalise the frame (translation only) -----------------------------
    # A translation changes nothing physical -- geometry, load direction and clamps are
    # all unchanged, only the coordinates shift -- so removing it is free and makes
    # positions comparable across models.
    #
    # Centred on the five standardised interface landmarks, NOT the body centre of
    # gravity. The body COG legitimately moves with the design, so centring there would
    # push the bolt holes to different absolute positions in every model and destroy
    # the one regularity worth keeping.
    #
    # Rotation is deliberately NOT corrected here. The load is a fixed global vector
    # that does not rotate with the part, so re-orienting a model would change the
    # problem rather than the coordinates. The QA stage measures rotation across models
    # and excludes the outliers instead.
    landmarks = landmark_centroids(deck, points, spc_set, load_set)
    origin = landmarks.mean(axis=0)
    points_centred = points - origin

    # ---- reduce to the surface -------------------------------------------------
    surface_ids = np.flatnonzero(is_surface)          # global 0-based indices
    global_to_local = np.full(len(points), -1, dtype=np.int32)
    global_to_local[surface_ids] = np.arange(len(surface_ids), dtype=np.int32)

    pos = points_centred[surface_ids].astype(np.float32)

    edges_global = surface_edges(tets, is_surface)
    if edges_global.size == 0:
        raise GraphBuildError(model_id, "no edges between surface nodes")
    edge_index = global_to_local[edges_global].astype(np.int32)   # (E, 2), undirected

    faces_global = boundary_faces(tets)
    faces_local = global_to_local[faces_global]
    if (faces_local < 0).any():
        raise GraphBuildError(
            model_id, "a boundary face touches a node not flagged as surface"
        )
    normals = vertex_normals(points_centred[surface_ids], faces_local).astype(np.float32)

    # ---- node features ---------------------------------------------------------
    node_type = np.full(len(surface_ids), NODE_NORMAL, dtype=np.int8)
    node_type[global_to_local[fixed_ids - 1]] = NODE_FIXED
    node_type[global_to_local[loaded_ids - 1]] = NODE_LOADED

    # Distance to the nearest clamp and to the nearest loaded node. Stress flows from
    # where load enters to where the part is held, so proximity to each end of that
    # path is directly informative -- and it is a global cue that local message passing
    # would otherwise need many hops to recover.
    #
    # Euclidean, not geodesic: geodesic distance through the mesh would be more
    # faithful for a curved part but costs a shortest-path solve per model for a
    # modest gain.
    dist_fixed = cKDTree(pos[node_type == NODE_FIXED]).query(pos)[0].astype(np.float32)
    dist_loaded = cKDTree(pos[node_type == NODE_LOADED]).query(pos)[0].astype(np.float32)

    # ---- targets, all four load cases ------------------------------------------
    y = np.empty((len(surface_ids), len(LOAD_CASES), len(TARGET_CHANNELS)),
                 dtype=np.float32)
    for i, case in enumerate(LOAD_CASES):
        cols = [f"{case}_{ch}" for ch in TARGET_CHANNELS]
        y[:, i, :] = df.loc[is_surface, cols].to_numpy(dtype=np.float32)

    # Row order must survive the surface filter unchanged: y row k has to be the same
    # node as pos row k. Checked against the coordinates rather than assumed.
    csv_surface_xyz = df.loc[is_surface, ["x", "y", "z"]].to_numpy() - origin
    if np.abs(csv_surface_xyz - pos).max() > 1e-3:
        raise GraphBuildError(model_id, "target rows are misaligned with node positions")

    if not np.isfinite(y).all():
        raise GraphBuildError(model_id, "non-finite values in the result fields")

    load_vectors = np.stack(
        [deck.loads[s].vector for s in sorted(deck.loads)]
    ).astype(np.float32)

    return {
        # geometry
        "pos": torch.from_numpy(pos),
        "edge_index_undirected": torch.from_numpy(edge_index),
        "normals": torch.from_numpy(normals),
        # per-node features
        "node_type": torch.from_numpy(node_type),
        "dist_fixed": torch.from_numpy(dist_fixed),
        "dist_loaded": torch.from_numpy(dist_loaded),
        # targets: (n_nodes, 4 load cases, 4 channels)
        "y": torch.from_numpy(y),
        # graph-level, kept for the ablations and for QA
        "meta": {
            "model_id": int(model_id),
            "n_mesh_nodes": int(len(points)),
            "n_surface_nodes": int(len(surface_ids)),
            "n_tets": int(len(tets)),
            "n_edges_undirected": int(len(edge_index)),
            "surface_fraction": float(is_surface.mean()),
            "load_cases": list(LOAD_CASES),
            "target_channels": list(TARGET_CHANNELS),
            "load_vectors": load_vectors,
            "material": {
                "E": deck.material.E,
                "nu": deck.material.nu,
                "rho": deck.material.rho,
            },
            # Landmarks before centring, so the QA stage can measure rotation between
            # models. The translation applied is recorded so it is reversible.
            "landmarks_raw": landmarks.astype(np.float64),
            "origin": origin.astype(np.float64),
            "n_fixed_nodes": int(len(fixed_ids)),
            "n_loaded_nodes": int(len(loaded_ids)),
        },
    }


def save_graph(graph: dict, out_dir: str | Path) -> Path:
    """Write one cached graph to ``<out_dir>/<model_id>.pt``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{graph['meta']['model_id']}.pt"
    torch.save(graph, path)
    return path
