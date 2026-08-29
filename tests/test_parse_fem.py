"""Tests for the ``.fem`` deck parser, against SimJEB model 148.

The parser's job is to turn a 44 MB solver deck into a handful of facts that every
later stage depends on. Most of the failure modes here are silent -- a load vector
read a thousand times too small, or bolt nodes attached to the wrong indices, would
train to a plausible-looking loss curve and a meaningless model. So the assertions
are specific values, not just shapes.
"""

import numpy as np
import pytest

from src.data.parse_fem import (
    FemDeck,
    landmark_centroids,
    nastran_float,
    parse_fem,
)
from tests.conftest import FIXTURE_N_GRID, FIXTURE_N_MESH_NODES

# The four SimJEB load cases. Magnitudes are identical in all 381 decks -- verified by
# grepping the FORCE/MOMENT cards across the dataset -- so they are safe to assert.
EXPECTED_LOADS = {
    2: ("FORCE", np.array([0.0, 0.0, 35585.77])),
    3: ("FORCE", np.array([-37809.9, 0.0, 0.0])),
    4: ("FORCE", np.array([-28276.2, 0.0, 31403.9])),
    5: ("MOMENT", np.array([0.0, 0.0, 564924.2])),
}


class TestNastranFloat:
    """Nastran drops the E from scientific notation; misreading it is silent."""

    def test_plain_reals(self):
        assert nastran_float("113800.0") == 113800.0
        assert nastran_float("0.342") == 0.342
        assert nastran_float("-37809.9") == -37809.9
        assert nastran_float("  1.0   ") == 1.0

    def test_implicit_exponent(self):
        # The density field in every SimJEB deck. A lenient parser reads this as 4.43,
        # a factor of a billion out.
        assert nastran_float("4.43-9") == pytest.approx(4.43e-9)
        assert nastran_float("1.2+3") == pytest.approx(1200.0)
        assert nastran_float("-4.43-9") == pytest.approx(-4.43e-9)

    def test_explicit_exponent_is_left_alone(self):
        assert nastran_float("4.43E-9") == pytest.approx(4.43e-9)
        assert nastran_float("4.43e-9") == pytest.approx(4.43e-9)
        assert nastran_float("1.0D+3") == pytest.approx(1000.0)

    def test_blank_is_none(self):
        assert nastran_float("        ") is None
        assert nastran_float("") is None

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            nastran_float("not a number")


class TestDeckStructure:
    def test_node_counts(self, deck: FemDeck):
        assert deck.n_grid == FIXTURE_N_GRID
        # Inferred from where the rigid reference nodes start, not hard-coded.
        assert deck.n_mesh_nodes == FIXTURE_N_MESH_NODES

    def test_reference_nodes_sit_above_the_mesh(self, deck: FemDeck):
        # This is what licenses vtk_index = grid_id - 1 everywhere downstream.
        assert sorted(deck.ref_nodes) == list(range(129_261, 129_266))

    def test_validate_passes_against_the_real_mesh(self, deck: FemDeck):
        deck.validate(n_mesh_nodes=FIXTURE_N_MESH_NODES)

    def test_validate_catches_a_mesh_mismatch(self, deck: FemDeck):
        with pytest.raises(ValueError, match="wrong nodes"):
            deck.validate(n_mesh_nodes=FIXTURE_N_MESH_NODES + 1)


class TestMaterial:
    def test_ti6al4v(self, deck: FemDeck):
        mat = deck.material
        assert mat.mid == 1
        # 113800 MPa, i.e. 113.8 GPa. The dataset README labels this GPa while writing
        # N/mm^2 beside it; N/mm^2 is MPa, and the deck is the ground truth.
        assert mat.E == pytest.approx(113800.0)
        assert mat.nu == pytest.approx(0.342)
        assert mat.rho == pytest.approx(4.43e-9)

    def test_solid_property_points_at_that_material(self, deck: FemDeck):
        assert deck.property_mid == {1: 1}


class TestSubcases:
    def test_four_load_cases_in_order(self, deck: FemDeck):
        assert [s.label for s in deck.subcases] == [
            "vertical", "horizontal", "diagonal", "torsion"
        ]

    def test_every_subcase_shares_one_spc_set(self, deck: FemDeck):
        # The part is clamped the same way in all four cases; only the load changes.
        assert {s.spc_set for s in deck.subcases} == {1}

    def test_load_sets_are_distinct(self, deck: FemDeck):
        assert [s.load_set for s in deck.subcases] == [2, 3, 4, 5]

    def test_lookup_by_label(self, deck: FemDeck):
        assert deck.subcase_by_label("vertical").load_set == 2
        assert deck.subcase_by_label("VERTICAL").load_set == 2  # case-insensitive
        with pytest.raises(KeyError):
            deck.subcase_by_label("shear")


class TestLoads:
    @pytest.mark.parametrize("set_id", sorted(EXPECTED_LOADS))
    def test_load_vectors(self, deck: FemDeck, set_id: int):
        kind, vector = EXPECTED_LOADS[set_id]
        load = deck.loads[set_id]
        assert load.kind == kind
        np.testing.assert_allclose(load.vector, vector, rtol=1e-9)

    def test_all_loads_act_at_the_same_reference_node(self, deck: FemDeck):
        # All four load cases are applied at the load lug's RBE3 reference node.
        assert {ld.node for ld in deck.loads.values()} == {129_265}

    def test_diagonal_is_a_superposition_of_the_other_two(self, deck: FemDeck):
        # Linear-static analysis, so this is not a coincidence: the diagonal case adds
        # no independent loading direction. Worth asserting because it is the reason
        # diagonal was ruled out as a candidate single load case.
        ver = deck.loads[2].vector
        hor = deck.loads[3].vector
        dia = deck.loads[4].vector
        a = dia[0] / hor[0]
        b = dia[2] / ver[2]
        np.testing.assert_allclose(dia, a * hor + b * ver, rtol=1e-6)
        assert 0.7 < a < 0.8 and 0.85 < b < 0.9


class TestBoundaryConditions:
    def test_four_bolt_holes_are_constrained(self, deck: FemDeck):
        spc = deck.spcs[1]
        assert len(spc) == 4
        assert [node for node, _ in spc] == [129_261, 129_262, 129_263, 129_264]
        # 123456 = all six degrees of freedom, i.e. fully fixed.
        assert {dof for _, dof in spc} == {"123456"}

    def test_rigid_elements(self, deck: FemDeck):
        kinds = [r.kind for r in deck.rigids]
        assert kinds.count("RBE2") == 4   # one spider per bolt hole
        assert kinds.count("RBE3") == 1   # one distributing element at the load lug

    def test_fixed_nodes_expand_to_real_mesh_nodes(self, deck: FemDeck):
        fixed = deck.fixed_mesh_nodes(spc_set=1)
        assert fixed.size > 100, "an SPC that did not expand through the RBE2s"
        assert fixed.min() >= 1
        assert fixed.max() <= deck.n_mesh_nodes
        assert np.all(np.diff(fixed) > 0), "expected sorted, unique node ids"

    def test_loaded_nodes_expand_through_the_rbe3(self, deck: FemDeck):
        loaded = deck.loaded_mesh_nodes(load_set=2)
        assert loaded.size > 100
        assert loaded.max() <= deck.n_mesh_nodes

    def test_load_case_does_not_change_which_nodes_are_loaded(self, deck: FemDeck):
        # All four cases push on the same lug; only the direction differs.
        groups = [deck.loaded_mesh_nodes(s) for s in (2, 3, 4, 5)]
        for other in groups[1:]:
            np.testing.assert_array_equal(groups[0], other)

    def test_fixed_and_loaded_sets_are_disjoint(self, deck: FemDeck):
        fixed = set(deck.fixed_mesh_nodes(1).tolist())
        loaded = set(deck.loaded_mesh_nodes(2).tolist())
        assert not (fixed & loaded), "a node cannot be both clamped and loaded"

    def test_four_separate_bolt_groups(self, deck: FemDeck):
        groups = deck.bolt_groups(spc_set=1)
        assert len(groups) == 4
        assert all(g.size > 20 for g in groups)
        # Distinct bolt holes must not share nodes.
        seen: set[int] = set()
        for g in groups:
            assert not (seen & set(g.tolist()))
            seen |= set(g.tolist())


class TestInterfaceLandmarks:
    def test_five_interfaces(self, deck: FemDeck):
        groups = deck.interface_groups(spc_set=1, load_set=2)
        assert len(groups) == 5  # 4 bolt holes + 1 load lug

    def test_centroids_are_distinct_points(self, deck: FemDeck, csv_path):
        import pandas as pd

        coords = pd.read_csv(csv_path, usecols=["x", "y", "z"]).to_numpy()
        assert coords.shape[0] == deck.n_mesh_nodes

        landmarks = landmark_centroids(deck, coords, spc_set=1, load_set=2)
        assert landmarks.shape == (5, 3)
        assert np.isfinite(landmarks).all()

        # The five interfaces are spread across the bracket, so no two centroids
        # should coincide -- if they did, the alignment check would be degenerate.
        for i in range(5):
            for j in range(i + 1, 5):
                assert np.linalg.norm(landmarks[i] - landmarks[j]) > 1.0


class TestGridCoordinates:
    """The optional GRID pass, used by QA to confirm the deck and the CSV agree."""

    def test_grid_coords_match_the_result_csv(self, fem_path, csv_path):
        import pandas as pd

        deck = parse_fem(fem_path, parse_grids=True)
        coords = pd.read_csv(csv_path, usecols=["id", "x", "y", "z"])

        # The load-bearing assumption of the whole pipeline: CSV row i is mesh node i,
        # and GRID id i is the same node. Checked on a spread of nodes rather than all
        # 129k, which is enough to catch an off-by-one or a renumbering.
        sample = np.linspace(1, deck.n_mesh_nodes, 200, dtype=int)
        for gid in sample:
            row = coords.iloc[gid - 1]
            assert int(row["id"]) == gid
            np.testing.assert_allclose(
                deck.grid_coords[gid],
                [row["x"], row["y"], row["z"]],
                rtol=1e-4,
                atol=1e-4,
            )
