"""Tests for the dataset layer: feature assembly, targets, and normalisation.

Builds one real cached graph from the fixture and writes it to a temporary directory,
so the tests exercise the same load path training will use.
"""

import numpy as np
import pytest
import torch

from src.data.build_graph import NODE_FIXED, NODE_LOADED, build_graph, save_graph
from src.data.dataset import (
    BASE_FEATURE_DIM,
    FeatureConfig,
    SimJEBDataset,
    build_data,
    edge_attributes,
    stress_forward,
    stress_inverse,
    to_bidirectional,
)
from src.data.normalization import (
    STD_FLOOR,
    NormalizationStats,
    Normalizer,
    compute_normalization_stats,
)
from tests.conftest import FIXTURE_ID


@pytest.fixture(scope="module")
def cached_graph(vtk_path, csv_path, fem_path, deck):
    return build_graph(FIXTURE_ID, vtk_path, csv_path, fem_path, deck=deck)


@pytest.fixture(scope="module")
def cache_dir(cached_graph, tmp_path_factory):
    d = tmp_path_factory.mktemp("graphs")
    save_graph(cached_graph, d)
    return d


class TestStressTransform:
    def test_round_trip(self):
        stress = torch.tensor([0.0, 1.0, 250.0, 880.0, 14902.0])
        torch.testing.assert_close(stress_inverse(stress_forward(stress)), stress,
                                   rtol=1e-5, atol=1e-3)

    def test_compresses_the_singular_tail(self):
        """The reason the transform exists: raw stress spans ~50x across models, and a
        handful of singular corner nodes would otherwise dominate an MSE gradient."""
        bulk, singular = torch.tensor([200.0]), torch.tensor([14902.0])
        raw_ratio = (singular / bulk).item()
        log_ratio = (stress_forward(singular) / stress_forward(bulk)).item()
        assert raw_ratio > 70
        assert log_ratio < 2.0

    def test_monotonic(self):
        # Order must be preserved, or the model is learning a different ranking.
        stress = torch.linspace(0, 15000, 500)
        assert torch.all(torch.diff(stress_forward(stress)) > 0)

    def test_zero_maps_to_zero(self):
        assert stress_forward(torch.tensor([0.0])).item() == 0.0


class TestEdgeConstruction:
    def test_bidirectional_doubles_and_mirrors(self):
        undirected = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
        e = to_bidirectional(undirected)
        assert e.shape == (2, 4)
        pairs = {tuple(p) for p in e.t().tolist()}
        assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1)}

    def test_edge_attr_is_delta_and_length(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        e = torch.tensor([[0], [1]])
        attr = edge_attributes(pos, e)
        torch.testing.assert_close(attr, torch.tensor([[3.0, 4.0, 0.0, 5.0]]))

    def test_edge_attr_is_antisymmetric(self):
        """i->j and j->i must carry opposite vectors but the same length. If they did
        not, the graph would encode a direction that does not exist in the geometry."""
        pos = torch.randn(6, 3)
        e = torch.tensor([[0, 1], [1, 0]])
        attr = edge_attributes(pos, e)
        torch.testing.assert_close(attr[0, :3], -attr[1, :3])
        torch.testing.assert_close(attr[0, 3], attr[1, 3])

    def test_edge_attr_on_the_real_graph(self, cached_graph):
        pos = cached_graph["pos"]
        e = to_bidirectional(cached_graph["edge_index_undirected"])
        attr = edge_attributes(pos, e)
        assert attr.shape == (e.shape[1], 4)
        assert torch.isfinite(attr).all()
        assert float(attr[:, 3].min()) > 0, "a zero-length edge means duplicate nodes"


class TestFeatureAssembly:
    def test_default_feature_dim(self, cached_graph):
        data = build_data(cached_graph, FeatureConfig())
        assert data.x.shape[1] == BASE_FEATURE_DIM == 8

    def test_material_flag_adds_three(self, cached_graph):
        data = build_data(cached_graph, FeatureConfig(use_material=True))
        assert data.x.shape[1] == 11

    def test_position_flag_adds_three(self, cached_graph):
        data = build_data(cached_graph, FeatureConfig(use_position=True))
        assert data.x.shape[1] == 11

    def test_material_columns_are_constant(self, cached_graph):
        """Pins the reason the material ablation is expected to be a null result: the
        columns carry no information, so the network can only learn to ignore them."""
        data = build_data(cached_graph, FeatureConfig(use_material=True))
        material_cols = data.x[:, BASE_FEATURE_DIM:]
        assert float(material_cols.std(dim=0).max()) == 0.0

    def test_node_type_one_hot(self, cached_graph):
        data = build_data(cached_graph, FeatureConfig())
        one_hot = data.x[:, :3]
        torch.testing.assert_close(one_hot.sum(dim=1), torch.ones(one_hot.shape[0]))
        node_type = cached_graph["node_type"].long()
        assert int(one_hot[:, NODE_FIXED].sum()) == int((node_type == NODE_FIXED).sum())
        assert int(one_hot[:, NODE_LOADED].sum()) == int((node_type == NODE_LOADED).sum())

    def test_features_are_finite(self, cached_graph):
        data = build_data(cached_graph, FeatureConfig(use_material=True,
                                                     use_position=True))
        assert torch.isfinite(data.x).all()


class TestFrameInvariance:
    """The default feature set is translation-invariant but NOT rotation-invariant.

    That asymmetry is deliberate and it is what justifies how frame outliers are
    handled: translation is normalised away as a mere change of coordinates, while a
    rotated model is excluded because it is a different physical problem.

    The reason rotation must not be invariant: the load is a fixed global vector,
    identical in all 381 decks, and it is not fed to the model precisely because it is
    constant. So the network learns implicitly that load arrives from +Z. Rotate a
    bracket and the same global load meets a different face -- same shape, different
    physics, with nothing in the features to signal the change.

    If a future feature breaks either property, it should fail here rather than quietly
    change what the model is capable of learning.
    """

    @staticmethod
    def _translated(cached, shift):
        moved = dict(cached)
        moved["pos"] = cached["pos"] + torch.tensor(shift)
        return moved

    @staticmethod
    def _rotated_z(cached, radians):
        c, s = float(np.cos(radians)), float(np.sin(radians))
        R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        turned = dict(cached)
        turned["pos"] = cached["pos"] @ R.T
        turned["normals"] = cached["normals"] @ R.T
        return turned

    def test_features_survive_a_large_translation(self, cached_graph):
        """Move the whole bracket a kilometre; the model must see identical input."""
        base = build_data(cached_graph, FeatureConfig())
        moved = build_data(self._translated(cached_graph, [1000.0, 500.0, -250.0]),
                           FeatureConfig())

        torch.testing.assert_close(moved.x, base.x)
        torch.testing.assert_close(moved.edge_attr, base.edge_attr, atol=1e-3, rtol=0)
        torch.testing.assert_close(moved.y, base.y)

    def test_translation_does_move_the_coordinates(self, cached_graph):
        """Guards the test above from passing vacuously."""
        base = build_data(cached_graph, FeatureConfig())
        moved = build_data(self._translated(cached_graph, [1000.0, 0.0, 0.0]),
                           FeatureConfig())
        assert float((moved.pos - base.pos).abs().max()) == pytest.approx(1000.0)

    def test_position_feature_breaks_translation_invariance(self, cached_graph):
        """The one feature that would make the model care where a bracket sits.

        Documents the cost of ``use_position=True``: the network could then key on
        absolute location, which does not transfer to an unseen design.
        """
        cfg = FeatureConfig(use_position=True)
        base = build_data(cached_graph, cfg)
        moved = build_data(self._translated(cached_graph, [1000.0, 0.0, 0.0]), cfg)
        assert not torch.allclose(moved.x, base.x)

    def test_features_change_under_rotation(self, cached_graph):
        """Rotation must NOT be invisible -- the load frame does not rotate with the
        part, so a turned bracket is a different problem, not a different view."""
        base = build_data(cached_graph, FeatureConfig())
        turned = build_data(self._rotated_z(cached_graph, np.pi / 2), FeatureConfig())

        assert not torch.allclose(turned.x, base.x), "rotation was silently absorbed"
        assert not torch.allclose(turned.edge_attr, base.edge_attr)

    def test_edge_lengths_are_preserved_by_rotation(self, cached_graph):
        """Rotation is still rigid: directions turn, distances do not. If lengths
        changed, the rotation matrix itself would be wrong."""
        base = build_data(cached_graph, FeatureConfig())
        turned = build_data(self._rotated_z(cached_graph, np.pi / 2), FeatureConfig())
        torch.testing.assert_close(turned.edge_attr[:, 3], base.edge_attr[:, 3],
                                   atol=1e-3, rtol=0)


class TestTargets:
    def test_stress_only_by_default(self, cached_graph):
        data = build_data(cached_graph, FeatureConfig())
        assert data.y.shape[1] == 1

    def test_aux_displacement_gives_four_channels(self, cached_graph):
        data = build_data(cached_graph, FeatureConfig(use_aux_displacement=True))
        assert data.y.shape[1] == 4
        # Channel 0 stays stress, so the primary objective is unchanged.
        plain = build_data(cached_graph, FeatureConfig())
        torch.testing.assert_close(data.y[:, :1], plain.y)

    def test_log_transform_applied(self, cached_graph):
        logged = build_data(cached_graph, FeatureConfig(log_stress=True))
        raw = build_data(cached_graph, FeatureConfig(log_stress=False))
        torch.testing.assert_close(stress_inverse(logged.y), raw.y, rtol=1e-4, atol=1e-2)

    def test_raw_mpa_is_kept_alongside(self, cached_graph):
        """Evaluation must score in MPa; carrying the untransformed values avoids a
        second pass over the cache to recover them."""
        data = build_data(cached_graph, FeatureConfig(log_stress=True))
        torch.testing.assert_close(stress_inverse(data.y), data.stress_mpa,
                                   rtol=1e-4, atol=1e-2)

    def test_load_case_selection_changes_the_target(self, cached_graph):
        ver = build_data(cached_graph, FeatureConfig(load_case="ver")).y
        tor = build_data(cached_graph, FeatureConfig(load_case="tor")).y
        assert not torch.allclose(ver, tor)

    def test_unknown_load_case_raises(self, cached_graph):
        with pytest.raises(ValueError, match="unknown load case"):
            build_data(cached_graph, FeatureConfig(load_case="shear"))


class TestDataset:
    def test_loads_from_disk(self, cache_dir):
        ds = SimJEBDataset(cache_dir, [FIXTURE_ID])
        assert len(ds) == 1
        data = ds[0]
        assert data.model_id == FIXTURE_ID
        assert data.x.shape[0] == data.pos.shape[0] == data.y.shape[0]

    def test_missing_model_raises_immediately(self, cache_dir):
        # Better a clear error at construction than a crash mid-epoch.
        with pytest.raises(FileNotFoundError, match="cached graphs missing"):
            SimJEBDataset(cache_dir, [FIXTURE_ID, 99_999])

    def test_batching(self, cache_dir):
        from torch_geometric.loader import DataLoader

        ds = SimJEBDataset(cache_dir, [FIXTURE_ID, FIXTURE_ID])
        batch = next(iter(DataLoader(ds, batch_size=2)))
        n = ds[0].x.shape[0]
        assert batch.x.shape[0] == 2 * n
        assert int(batch.batch.max()) == 1
        # PyG must offset the second graph's edge indices, not overlay them.
        assert int(batch.edge_index.max()) == 2 * n - 1


class TestNormalization:
    def test_normalize_round_trip(self):
        norm = Normalizer(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        x = torch.randn(50, 2)
        torch.testing.assert_close(norm.denormalize(norm.normalize(x)), x)

    def test_fit_recovers_mean_and_std(self):
        x = torch.randn(5000, 3) * torch.tensor([2.0, 5.0, 0.1]) + 3.0
        norm = Normalizer.from_accumulator(
            x.shape[0], x.double().sum(0), x.double().pow(2).sum(0)
        )
        torch.testing.assert_close(norm.mean.float(), x.mean(0), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(norm.std.float(), x.std(0, unbiased=False),
                                   rtol=1e-3, atol=1e-3)

    def test_constant_channel_does_not_explode(self):
        """The failure this guards: a constant column -- material features, or a
        one-hot that is all zeros in a small split -- would divide by zero and take
        the whole run down with NaNs."""
        x = torch.cat([torch.randn(200, 1), torch.full((200, 1), 7.0)], dim=1)
        norm = Normalizer.from_accumulator(
            x.shape[0], x.double().sum(0), x.double().pow(2).sum(0)
        )
        # approx, not exact: statistics are stored in float32 to match the model's
        # weights, and 1e-6 is not exactly representable there.
        assert float(norm.std[1]) == pytest.approx(STD_FLOOR, rel=1e-5)
        assert torch.isfinite(norm.normalize(x)).all()

    def test_statistics_are_float32(self):
        """Fitting accumulates in float64 for stability, but the stored statistics must
        match the model's dtype -- float64 stats silently promote the activations and
        fail inside the first Linear layer."""
        x = torch.randn(100, 3)
        norm = Normalizer.from_accumulator(
            x.shape[0], x.double().sum(0), x.double().pow(2).sum(0)
        )
        assert norm.mean.dtype == torch.float32
        assert norm.normalize(x).dtype == torch.float32

    def test_stats_fitted_on_the_real_dataset(self, cache_dir):
        ds = SimJEBDataset(cache_dir, [FIXTURE_ID])
        stats = compute_normalization_stats(ds)
        data = ds[0]
        assert stats.node.mean.shape == (data.x.shape[1],)
        assert stats.edge.mean.shape == (data.edge_attr.shape[1],)
        assert stats.target.mean.shape == (data.y.shape[1],)
        assert torch.isfinite(stats.node.mean).all()
        assert bool((stats.node.std > 0).all())

    def test_normalized_features_are_centred(self, cache_dir):
        ds = SimJEBDataset(cache_dir, [FIXTURE_ID])
        stats = compute_normalization_stats(ds)
        normalized = stats.node.normalize(ds[0].x.double())
        # Channels with real variance should come out ~zero-mean, ~unit-variance.
        varying = stats.node.std > 1e-3
        torch.testing.assert_close(
            normalized[:, varying].mean(0),
            torch.zeros(int(varying.sum()), dtype=torch.float64),
            atol=1e-4, rtol=0,
        )

    def test_state_dict_round_trip(self, cache_dir):
        ds = SimJEBDataset(cache_dir, [FIXTURE_ID])
        stats = compute_normalization_stats(ds)
        restored = NormalizationStats.from_state_dict(stats.state_dict())
        torch.testing.assert_close(restored.node.mean, stats.node.mean)
        torch.testing.assert_close(restored.target.std, stats.target.std)

    def test_empty_dataset_raises(self, cache_dir):
        with pytest.raises(ValueError, match="empty"):
            compute_normalization_stats(SimJEBDataset(cache_dir, []))
