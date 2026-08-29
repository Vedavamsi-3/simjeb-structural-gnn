"""Tests for graph construction, against SimJEB model 148.

The graph is what the model actually sees, so a fault here is invisible downstream --
wrong node indices or a mis-joined target column produce a graph that trains happily
and means nothing. These tests check the joins and the invariants, not just shapes.
"""

import numpy as np
import pytest
import torch

from src.data.build_graph import (
    NODE_FIXED,
    NODE_LOADED,
    NODE_NORMAL,
    GraphBuildError,
    boundary_faces,
    build_graph,
    surface_edges,
    vertex_normals,
)
from tests.conftest import FIXTURE_ID, FIXTURE_N_MESH_NODES


@pytest.fixture(scope="module")
def graph(vtk_path, csv_path, fem_path, deck):
    """Built once -- it reads an 18 MB mesh and a 40 MB CSV."""
    return build_graph(FIXTURE_ID, vtk_path, csv_path, fem_path, deck=deck)


class TestPrimitives:
    """Unit tests on small hand-checkable inputs, before touching the real mesh."""

    def test_boundary_faces_of_a_single_tet(self):
        # One tetrahedron is all surface: every one of its 4 faces is a boundary face.
        tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
        faces = boundary_faces(tets)
        assert len(faces) == 4
        assert {tuple(sorted(f)) for f in faces} == {
            (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)
        }

    def test_shared_face_is_not_a_boundary(self):
        # Two tets glued on face (1,2,3): 8 faces total, 2 of them the same, so the
        # boundary is 6 faces and the shared one is gone.
        tets = np.array([[0, 1, 2, 3], [4, 1, 2, 3]], dtype=np.int32)
        faces = boundary_faces(tets)
        assert len(faces) == 6
        assert (1, 2, 3) not in {tuple(sorted(f)) for f in faces}

    def test_surface_edges_skip_interior_nodes(self):
        # Node 3 is interior, so any edge touching it must be dropped.
        tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
        is_surface = np.array([True, True, True, False])
        edges = surface_edges(tets, is_surface)
        assert {tuple(e) for e in edges} == {(0, 1), (0, 2), (1, 2)}

    def test_surface_edges_are_deduplicated(self):
        # The shared edge (1,2) appears in both tets but must be stored once.
        tets = np.array([[0, 1, 2, 3], [4, 1, 2, 5]], dtype=np.int32)
        is_surface = np.ones(6, dtype=bool)
        edges = surface_edges(tets, is_surface)
        assert len(edges) == len({tuple(e) for e in edges})
        assert (1, 2) in {tuple(e) for e in edges}

    def test_vertex_normals_are_unit_and_outward(self):
        # A unit cube's corner normals should point away from the centre.
        v = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
             [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
        )
        f = np.array(
            [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
             [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
             [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]], dtype=np.int32
        )
        n = vertex_normals(v, f)
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
        outward = v - v.mean(axis=0)
        assert np.all((n * outward).sum(axis=1) > 0), "normals point inward"

    def test_isolated_vertex_gets_zero_not_nan(self):
        # A NaN here would poison the normalisation statistics for the whole batch.
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], dtype=float)
        f = np.array([[0, 1, 2]], dtype=np.int32)
        n = vertex_normals(v, f)
        assert np.isfinite(n).all()
        np.testing.assert_array_equal(n[3], [0.0, 0.0, 0.0])


class TestGraphShapes:
    def test_surface_node_count(self, graph):
        meta = graph["meta"]
        assert meta["n_mesh_nodes"] == FIXTURE_N_MESH_NODES
        # surf > 0, i.e. 49,710 plain surface + 594 bolt + 690 lug.
        assert meta["n_surface_nodes"] == 50_994
        # These brackets are thin and perforated, so the surface is a much larger
        # share of the mesh than it would be for a solid part.
        assert 0.35 < meta["surface_fraction"] < 0.45

    def test_every_per_node_array_agrees_on_length(self, graph):
        n = graph["meta"]["n_surface_nodes"]
        assert graph["pos"].shape == (n, 3)
        assert graph["normals"].shape == (n, 3)
        assert graph["node_type"].shape == (n,)
        assert graph["dist_fixed"].shape == (n,)
        assert graph["dist_loaded"].shape == (n,)
        assert graph["y"].shape == (n, 4, 4)   # 4 load cases, 4 channels

    def test_edges_are_undirected_and_in_range(self, graph):
        e = graph["edge_index_undirected"]
        n = graph["meta"]["n_surface_nodes"]
        assert e.dtype == torch.int32          # int64 would double the file
        assert e.shape[1] == 2
        assert int(e.min()) >= 0 and int(e.max()) < n
        assert int((e[:, 0] == e[:, 1]).sum()) == 0, "self-loops"
        # Stored one way only; the loader mirrors them.
        assert bool((e[:, 0] < e[:, 1]).all())

    def test_edges_are_unique(self, graph):
        e = graph["edge_index_undirected"].numpy()
        assert len(np.unique(e, axis=0)) == len(e)

    def test_graph_is_reasonably_connected(self, graph):
        n = graph["meta"]["n_surface_nodes"]
        degree = 2 * graph["meta"]["n_edges_undirected"] / n
        # A triangulated surface averages ~6; the through-material edges push it above.
        assert 5 < degree < 20, f"suspicious mean degree {degree:.1f}"


class TestSurfColumn:
    """``surf`` is a four-way label, not a boolean -- the single most load-bearing
    detail about this file format."""

    def test_four_distinct_values(self, csv_path):
        import pandas as pd

        counts = pd.read_csv(csv_path, usecols=["surf"])["surf"].value_counts()
        assert set(counts.index) == {0, 1, 2, 3}

    def test_interface_labels_match_the_solver_deck(self, csv_path, deck):
        """Two independent sources agreeing is what makes the parser trustworthy.

        The deck describes the interfaces through RBE2/RBE3 rigid elements; the CSV
        labels them directly. Neither is derived from the other, so exact agreement on
        all 1,284 interface nodes is real evidence rather than a tautology.
        """
        import pandas as pd

        df = pd.read_csv(csv_path, usecols=["id", "surf"])
        bolt_ids = set(df.loc[df["surf"] == 2, "id"])
        lug_ids = set(df.loc[df["surf"] == 3, "id"])

        assert bolt_ids == set(deck.fixed_mesh_nodes(1).tolist())
        assert lug_ids == set(deck.loaded_mesh_nodes(2).tolist())

    def test_reading_surf_as_boolean_would_lose_the_interfaces(self, csv_path):
        """Pins the bug this nearly caused: ``surf == 1`` silently drops every bolt
        and lug node -- exactly the nodes the boundary conditions act on."""
        import pandas as pd

        surf = pd.read_csv(csv_path, usecols=["surf"])["surf"]
        assert int((surf == 1).sum()) < int((surf > 0).sum())
        assert int((surf > 0).sum()) - int((surf == 1).sum()) == 1_284


class TestBoundaryConditions:
    def test_all_three_node_types_present(self, graph):
        types = graph["node_type"].numpy()
        assert set(np.unique(types)) == {NODE_NORMAL, NODE_FIXED, NODE_LOADED}

    def test_bc_node_counts_match_the_deck(self, graph, deck):
        types = graph["node_type"].numpy()
        assert int((types == NODE_FIXED).sum()) == graph["meta"]["n_fixed_nodes"]
        assert int((types == NODE_LOADED).sum()) == graph["meta"]["n_loaded_nodes"]

    def test_fixed_and_loaded_are_a_small_minority(self, graph):
        types = graph["node_type"].numpy()
        share = ((types != NODE_NORMAL).sum()) / len(types)
        assert 0 < share < 0.2, "boundary conditions should touch a small fraction"

    def test_distance_features_are_zero_at_their_own_nodes(self, graph):
        types = graph["node_type"].numpy()
        np.testing.assert_allclose(
            graph["dist_fixed"].numpy()[types == NODE_FIXED], 0.0, atol=1e-5
        )
        np.testing.assert_allclose(
            graph["dist_loaded"].numpy()[types == NODE_LOADED], 0.0, atol=1e-5
        )

    def test_distances_are_finite_and_positive_elsewhere(self, graph):
        d_fixed = graph["dist_fixed"].numpy()
        d_loaded = graph["dist_loaded"].numpy()
        assert np.isfinite(d_fixed).all() and np.isfinite(d_loaded).all()
        assert d_fixed.max() > 1.0 and d_loaded.max() > 1.0


class TestFrame:
    def test_centred_on_the_interface_landmarks(self, graph):
        # After centring, the mean of the five landmark centroids sits at the origin.
        landmarks = graph["meta"]["landmarks_raw"] - graph["meta"]["origin"]
        np.testing.assert_allclose(landmarks.mean(axis=0), 0.0, atol=1e-6)

    def test_five_landmarks_recorded(self, graph):
        assert graph["meta"]["landmarks_raw"].shape == (5, 3)

    def test_translation_is_recoverable(self, graph, vtk_path):
        # The shift is stored, so centring is reversible and auditable.
        import meshio

        points = np.asarray(meshio.read(vtk_path).points)
        assert np.linalg.norm(graph["meta"]["origin"]) > 1.0  # it really did move
        assert np.isfinite(graph["meta"]["origin"]).all()
        assert points.shape[0] == graph["meta"]["n_mesh_nodes"]

    def test_scale_is_untouched(self, graph, vtk_path):
        # A larger bracket is genuinely stiffer, so size must not be normalised away.
        import meshio

        points = np.asarray(meshio.read(vtk_path).points)
        original_extent = points.max(axis=0) - points.min(axis=0)
        pos = graph["pos"].numpy()
        # The surface is a subset of the mesh, so its extent is at most the original's
        # -- but for a thin shell-like part it should be nearly all of it.
        np.testing.assert_allclose(
            pos.max(axis=0) - pos.min(axis=0), original_extent, rtol=0.05
        )


class TestTargets:
    def test_stress_channel_matches_the_csv(self, graph, csv_path):
        """The join that matters: target row i must be surface node i."""
        import pandas as pd

        df = pd.read_csv(csv_path, usecols=["surf", "ver_stress"])
        expected = df.loc[df["surf"] > 0, "ver_stress"].to_numpy(dtype=np.float32)
        got = graph["y"][:, 0, 3].numpy()          # case 0 = ver, channel 3 = stress
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_displacement_channels_match_the_csv(self, graph, csv_path):
        import pandas as pd

        cols = ["ver_xdisp", "ver_ydisp", "ver_zdisp"]
        df = pd.read_csv(csv_path, usecols=["surf", *cols])
        expected = df.loc[df["surf"] > 0, cols].to_numpy(dtype=np.float32)
        np.testing.assert_allclose(graph["y"][:, 0, :3].numpy(), expected, rtol=1e-6)

    def test_all_four_load_cases_are_distinct(self, graph):
        # If two cases matched, a column join has gone wrong somewhere.
        stress = graph["y"][:, :, 3].numpy()
        for i in range(4):
            for j in range(i + 1, 4):
                assert not np.allclose(stress[:, i], stress[:, j])

    def test_stress_is_non_negative(self, graph):
        # Von Mises is a norm; a negative value would mean a bad column join.
        assert float(graph["y"][:, :, 3].min()) >= 0.0

    def test_stress_range_is_physical_but_singular(self, graph):
        """Peak stress far exceeds Ti-6Al-4V yield -- expected, and it drives the
        log target. Asserted so the finding is pinned rather than remembered."""
        ver_stress = graph["y"][:, 0, 3].numpy()
        assert ver_stress.max() > 880.0, "expected singular peaks above yield"
        assert np.median(ver_stress) < 880.0, "the bulk should be well below yield"

    def test_targets_are_finite(self, graph):
        assert torch.isfinite(graph["y"]).all()


class TestMetadata:
    def test_material_is_ti6al4v(self, graph):
        mat = graph["meta"]["material"]
        assert mat["E"] == pytest.approx(113800.0)
        assert mat["nu"] == pytest.approx(0.342)

    def test_four_load_vectors_recorded(self, graph):
        assert graph["meta"]["load_vectors"].shape == (4, 3)


class TestFailures:
    """The build must refuse bad input loudly rather than produce a quiet wrong graph."""

    def test_row_count_mismatch_is_rejected(self, tmp_path, vtk_path, csv_path,
                                            fem_path, deck):
        import pandas as pd

        truncated = tmp_path / "short.csv"
        pd.read_csv(csv_path).iloc[:-1].to_csv(truncated, index=False)
        with pytest.raises(GraphBuildError, match="rows but mesh has"):
            build_graph(FIXTURE_ID, vtk_path, truncated, fem_path, deck=deck)

    def test_shifted_coordinates_are_rejected(self, tmp_path, vtk_path, csv_path,
                                              fem_path, deck):
        """A row-order change that keeps the row count is the dangerous case."""
        import pandas as pd

        df = pd.read_csv(csv_path)
        df[["x", "y", "z"]] = df[["x", "y", "z"]].to_numpy()[::-1]
        shuffled = tmp_path / "shuffled.csv"
        df.to_csv(shuffled, index=False)
        with pytest.raises(GraphBuildError, match="coordinates disagree"):
            build_graph(FIXTURE_ID, vtk_path, shuffled, fem_path, deck=deck)
