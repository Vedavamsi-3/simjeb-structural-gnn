"""Parse the boundary conditions, loads and material out of a SimJEB ``.fem`` deck.

The ``.fem`` files are OptiStruct/Nastran bulk-data decks written by HyperMesh. They
are the only place the *physics inputs* live: the ``.vtk`` files carry geometry alone
and the ``field.csv`` files carry results, but neither says where the part is clamped
or how hard it is pushed.

What this module extracts, and why each piece is needed:

``MAT1``
    Young's modulus, Poisson's ratio, density. Constant across all 381 SimJEB models,
    so it is not a useful learned feature -- but it is parsed anyway so the material
    ablation can switch it on, and so a future multi-material dataset needs no changes.

``SUBCASE`` blocks
    Map each named load case (vertical / horizontal / diagonal / torsion) to the SPC
    set that constrains it and the LOAD set that drives it.

``SPC`` + ``RBE2``
    The four bolt holes. ``SPC`` fixes an abstract *reference* node; the ``RBE2`` rigid
    element ties that reference node to the ring of real mesh nodes around a bolt hole.
    Expanding the RBE2 is what turns "node 129261 is fixed" into "these 500 surface
    nodes are fixed", which is what the model actually needs as a feature.

``FORCE`` / ``MOMENT`` + ``RBE3``
    The load lug, by the same mechanism: the load is applied at a reference node that
    an ``RBE3`` distributes onto real mesh nodes.

The reference nodes are also used for a second purpose -- the centroid of each of the
five interface groups (4 bolts + 1 lug) gives a set of landmarks that are standardised
across all 381 models, which is what makes the alignment check in ``src/qa`` possible.

Node numbering
--------------
GRID ids are 1-based and the mesh nodes come first, with the rigid-element reference
nodes appended above them. Verified on the fixture: 129,265 GRID cards for a mesh of
129,260 points, and the 5 reference nodes are 129261-129265. So::

    vtk_index = grid_id - 1        for grid_id <= n_mesh_nodes

``n_mesh_nodes`` is inferred as ``min(reference node ids) - 1`` rather than assumed,
and :meth:`FemDeck.validate` checks it against the mesh the caller actually loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Nastran small-field format: an 8-character field, 9 fields to a line. Field 0 is the
# card name; fields 1-8 are data. Continuation lines repeat the pattern with a marker
# in field 0.
FIELD_WIDTH = 8
N_FIELDS = 9


def _fields(line: str) -> list[str]:
    """Split a fixed-format Nastran line into its 9 raw 8-character fields."""
    line = line.rstrip("\n").rstrip("\r")
    return [line[i * FIELD_WIDTH:(i + 1) * FIELD_WIDTH] for i in range(N_FIELDS)]


def _int(text: str) -> int | None:
    text = text.strip()
    return int(text) if text else None


def nastran_float(text: str) -> float | None:
    """Parse a Nastran real, including the implicit-exponent form.

    Nastran drops the ``E`` from scientific notation to save columns, so a density of
    4.43e-9 is written ``4.43-9``. A naive ``float()`` raises on that, and -- worse --
    a lenient parser that strips the sign would silently read it as 4.43, a billion
    times too large.
    """
    text = text.strip()
    if not text:
        return None

    text = text.replace("D", "E").replace("d", "e")  # Fortran double-precision marker
    try:
        return float(text)
    except ValueError:
        pass

    # Walk back from the end looking for a sign that is acting as an exponent marker,
    # i.e. one that is not the leading sign and not already preceded by an E.
    for i in range(len(text) - 1, 0, -1):
        if text[i] in "+-" and text[i - 1] not in "eE":
            return float(f"{text[:i]}e{text[i:]}")

    raise ValueError(f"cannot parse Nastran real from {text!r}")


@dataclass(frozen=True)
class Material:
    """A ``MAT1`` card -- linear elastic, isotropic."""

    mid: int
    E: float           # Young's modulus, MPa
    nu: float          # Poisson's ratio
    rho: float | None  # density, tonne/mm^3


@dataclass(frozen=True)
class Load:
    """A ``FORCE`` or ``MOMENT`` card, already scaled by its magnitude field."""

    set_id: int
    kind: str          # "FORCE" or "MOMENT"
    node: int          # the reference node it is applied at
    vector: np.ndarray # (3,) N for a force, N*mm for a moment


@dataclass(frozen=True)
class Subcase:
    """One ``SUBCASE`` block from the case control section."""

    sid: int
    label: str
    spc_set: int | None
    load_set: int | None


@dataclass
class RigidElement:
    """An ``RBE2`` or ``RBE3``: one reference node tied to many real mesh nodes."""

    kind: str             # "RBE2" or "RBE3"
    eid: int
    ref_node: int
    dofs: str             # e.g. "123456"
    nodes: list[int] = field(default_factory=list)


@dataclass
class FemDeck:
    """Everything this project needs from one ``.fem`` file."""

    path: Path
    n_grid: int
    material: Material | None
    property_mid: dict[int, int]
    subcases: list[Subcase]
    spcs: dict[int, list[tuple[int, str]]]  # spc set id -> [(node, dof string)]
    loads: dict[int, Load]                  # load set id -> Load
    rigids: list[RigidElement]
    grid_coords: dict[int, np.ndarray] | None = None  # only when parse_grids=True

    # ---- derived -------------------------------------------------------------

    @property
    def ref_nodes(self) -> set[int]:
        """Reference nodes of the rigid elements -- not part of the mesh."""
        return {r.ref_node for r in self.rigids}

    @property
    def n_mesh_nodes(self) -> int:
        """Number of real mesh nodes, i.e. GRID ids excluding the reference nodes.

        Inferred from the reference nodes rather than assumed, so a deck that numbers
        them differently is caught by :meth:`validate` instead of silently mis-indexing
        every target.
        """
        refs = self.ref_nodes
        if not refs:
            return self.n_grid
        return min(refs) - 1

    def subcase_by_label(self, label: str) -> Subcase:
        for sc in self.subcases:
            if sc.label.lower() == label.lower():
                return sc
        raise KeyError(f"no subcase labelled {label!r} in {self.path.name}; "
                       f"have {[s.label for s in self.subcases]}")

    def rigid_by_ref_node(self, node: int) -> RigidElement:
        for r in self.rigids:
            if r.ref_node == node:
                return r
        raise KeyError(f"no rigid element with reference node {node} in {self.path.name}")

    def fixed_mesh_nodes(self, spc_set: int) -> np.ndarray:
        """Mesh node ids held fixed by an SPC set, expanded through the RBE2 spiders.

        The SPC constrains abstract reference nodes; this walks each one out to the
        ring of real bolt-hole nodes it represents.
        """
        out: list[int] = []
        for node, _dof in self.spcs.get(spc_set, []):
            if node in self.ref_nodes:
                out.extend(self.rigid_by_ref_node(node).nodes)
            else:
                out.append(node)  # a directly constrained mesh node
        return np.unique(np.asarray(out, dtype=np.int64))

    def loaded_mesh_nodes(self, load_set: int) -> np.ndarray:
        """Mesh node ids the load is distributed onto, expanded through the RBE3."""
        load = self.loads[load_set]
        if load.node in self.ref_nodes:
            return np.unique(np.asarray(self.rigid_by_ref_node(load.node).nodes,
                                        dtype=np.int64))
        return np.asarray([load.node], dtype=np.int64)

    def bolt_groups(self, spc_set: int) -> list[np.ndarray]:
        """One array of mesh node ids per constrained bolt hole, in SPC card order."""
        groups = []
        for node, _dof in self.spcs.get(spc_set, []):
            if node in self.ref_nodes:
                groups.append(np.asarray(self.rigid_by_ref_node(node).nodes,
                                         dtype=np.int64))
        return groups

    def interface_groups(self, spc_set: int, load_set: int) -> list[np.ndarray]:
        """The standardised interfaces: the bolt groups, then the load lug.

        These are the parts SimJEB holds fixed across all 381 designs, so the centroid
        of each one is a landmark usable as a common reference frame. The bracket body
        is what varies and cannot serve that purpose.
        """
        return self.bolt_groups(spc_set) + [self.loaded_mesh_nodes(load_set)]

    def validate(self, n_mesh_nodes: int | None = None) -> None:
        """Assert the deck is internally consistent and matches the caller's mesh.

        Cheap, and it guards the failure mode that does not crash: if node numbering
        does not line up, every boundary condition attaches to the wrong node and the
        model trains happily on nonsense.
        """
        if self.material is None:
            raise ValueError(f"{self.path.name}: no MAT1 card found")
        if not self.subcases:
            raise ValueError(f"{self.path.name}: no SUBCASE blocks found")
        if not self.rigids:
            raise ValueError(f"{self.path.name}: no RBE2/RBE3 elements found")

        # Reference nodes must sit above the mesh nodes in a contiguous block.
        refs = sorted(self.ref_nodes)
        if refs != list(range(self.n_mesh_nodes + 1, self.n_grid + 1)):
            raise ValueError(
                f"{self.path.name}: rigid reference nodes {refs} are not the contiguous "
                f"block above the mesh nodes (n_grid={self.n_grid}). The "
                f"vtk_index = grid_id - 1 mapping cannot be trusted here."
            )

        # Every dependent node must be a real mesh node.
        for r in self.rigids:
            bad = [n for n in r.nodes if not 1 <= n <= self.n_mesh_nodes]
            if bad:
                raise ValueError(
                    f"{self.path.name}: {r.kind} {r.eid} references non-mesh nodes "
                    f"{bad[:5]}{'...' if len(bad) > 5 else ''}"
                )

        if n_mesh_nodes is not None and n_mesh_nodes != self.n_mesh_nodes:
            raise ValueError(
                f"{self.path.name}: deck implies {self.n_mesh_nodes} mesh nodes but the "
                f"mesh has {n_mesh_nodes}. Targets would attach to the wrong nodes."
            )


def parse_fem(path: str | Path, parse_grids: bool = False) -> FemDeck:
    """Read one ``.fem`` deck.

    Parameters
    ----------
    path
        The ``.fem`` file.
    parse_grids
        Also collect GRID coordinates. Off by default: the files are ~50 MB and
        almost entirely GRID and CTETRA cards, and node coordinates are available
        more cheaply from the result CSV. Switch it on for the QA cross-check that
        confirms the CSV and the deck agree.

    Notes
    -----
    Single pass, dispatching on the first character of each line. GRID and CTETRA
    account for ~99.9% of the lines and are rejected as early as possible.
    """
    path = Path(path)

    n_grid = 0
    material: Material | None = None
    property_mid: dict[int, int] = {}
    subcases: list[Subcase] = []
    spcs: dict[int, list[tuple[int, str]]] = {}
    loads: dict[int, Load] = {}
    rigids: list[RigidElement] = []
    grid_coords: dict[int, np.ndarray] | None = {} if parse_grids else None

    in_bulk = False
    pending: RigidElement | None = None  # rigid element accepting continuation lines

    # Case-control state: SUBCASE opens a block, the following indented lines fill it.
    cur_sid: int | None = None
    cur_label = ""
    cur_spc: int | None = None
    cur_load: int | None = None

    def close_subcase() -> None:
        nonlocal cur_sid, cur_label, cur_spc, cur_load
        if cur_sid is not None:
            subcases.append(Subcase(cur_sid, cur_label, cur_spc, cur_load))
        cur_sid, cur_label, cur_spc, cur_load = None, "", None, None

    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            if not raw or raw[0] == "$":  # comment
                continue

            if not in_bulk:
                stripped = raw.strip()
                if stripped.startswith("BEGIN BULK"):
                    close_subcase()
                    in_bulk = True
                elif stripped.startswith("SUBCASE"):
                    close_subcase()
                    cur_sid = _int(stripped[len("SUBCASE"):])
                elif stripped.startswith("LABEL"):
                    cur_label = stripped[len("LABEL"):].strip()
                elif stripped.startswith("SPC"):
                    cur_spc = _int(stripped.split("=", 1)[1]) if "=" in stripped else None
                elif stripped.startswith("LOAD"):
                    cur_load = _int(stripped.split("=", 1)[1]) if "=" in stripped else None
                continue

            c0 = raw[0]

            # --- the bulk of the file, rejected first -------------------------
            if c0 == "G":
                if raw.startswith("GRID"):
                    n_grid += 1
                    if grid_coords is not None:
                        f = _fields(raw)
                        gid = _int(f[1])
                        grid_coords[gid] = np.array(
                            [nastran_float(f[3]), nastran_float(f[4]), nastran_float(f[5])],
                            dtype=np.float64,
                        )
                    pending = None
                    continue
            elif c0 == "C":
                if raw.startswith(("CTETRA", "CHEXA", "CPENTA", "CTRIA", "CQUAD")):
                    pending = None
                    continue
            elif c0 == "+":
                # Continuation of the rigid element currently being built.
                if pending is not None:
                    pending.nodes.extend(
                        n for n in (_int(t) for t in _fields(raw)[1:]) if n is not None
                    )
                continue

            f = _fields(raw)
            card = f[0].strip()

            if card == "RBE2":
                # RBE2 | EID | GN (reference) | CM (dofs) | GM1 GM2 ... (dependent)
                pending = RigidElement(
                    kind="RBE2",
                    eid=_int(f[1]),
                    ref_node=_int(f[2]),
                    dofs=f[3].strip(),
                    nodes=[n for n in (_int(t) for t in f[4:]) if n is not None],
                )
                rigids.append(pending)

            elif card == "RBE3":
                # RBE3 | EID | (blank) | REFGRID | REFC | WT1 | C1 | G G ...
                pending = RigidElement(
                    kind="RBE3",
                    eid=_int(f[1]),
                    ref_node=_int(f[3]),
                    dofs=f[4].strip(),
                    nodes=[n for n in (_int(t) for t in f[7:]) if n is not None],
                )
                rigids.append(pending)

            else:
                pending = None  # any other card ends a continuation run

                if card == "SPC":
                    # SPC | SID | G | C | D
                    spcs.setdefault(_int(f[1]), []).append((_int(f[2]), f[3].strip()))

                elif card in ("FORCE", "MOMENT"):
                    # FORCE | SID | G | CID | F | N1 | N2 | N3   -> vector = F * N
                    scale = nastran_float(f[4]) or 0.0
                    direction = np.array(
                        [nastran_float(f[5]) or 0.0,
                         nastran_float(f[6]) or 0.0,
                         nastran_float(f[7]) or 0.0],
                        dtype=np.float64,
                    )
                    sid = _int(f[1])
                    loads[sid] = Load(sid, card, _int(f[2]), scale * direction)

                elif card == "MAT1":
                    # MAT1 | MID | E | G | NU | RHO
                    material = Material(
                        mid=_int(f[1]),
                        E=nastran_float(f[2]),
                        nu=nastran_float(f[4]),
                        rho=nastran_float(f[5]),
                    )

                elif card == "PSOLID":
                    property_mid[_int(f[1])] = _int(f[2])

                elif card == "ENDDATA":
                    break

    return FemDeck(
        path=path,
        n_grid=n_grid,
        material=material,
        property_mid=property_mid,
        subcases=subcases,
        spcs=spcs,
        loads=loads,
        rigids=rigids,
        grid_coords=grid_coords,
    )


def landmark_centroids(deck: FemDeck, coords: np.ndarray,
                       spc_set: int, load_set: int) -> np.ndarray:
    """Centroid of each standardised interface, as an (n_interfaces, 3) array.

    ``coords`` is the mesh node coordinates in VTK order, so node id ``i`` is row
    ``i - 1``. Order is the four bolt holes as they appear on the SPC card, then the
    load lug -- consistent across models because the decks are generated the same way,
    which is what lets the alignment check compare one model against another.
    """
    groups = deck.interface_groups(spc_set, load_set)
    return np.stack([coords[g - 1].mean(axis=0) for g in groups])
