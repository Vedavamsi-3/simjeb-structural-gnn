"""Tests for the training loop.

Runs a real (tiny) training job end to end on the single fixture model, so the loop,
the checkpointing and the resume path are all exercised rather than mocked. The
properties that matter here are the ones that only reveal themselves after hours of
GPU time otherwise: that a killed run resumes, that metrics are reported in MPa, and
that stopping conditions actually fire.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.build_graph import build_graph, save_graph
from src.data.split import Split
from src.train import (
    EpochRecord,
    TrainConfig,
    compute_loss,
    evaluate_split,
    plot_loss_curve,
    r2_score,
    set_seed,
    train,
)
from tests.conftest import FIXTURE_ID


@pytest.fixture(scope="module")
def workspace(vtk_path, csv_path, fem_path, deck, tmp_path_factory):
    """A miniature project on disk: one cached graph, and a split that reuses it for
    train, val and test. Enough to drive the whole loop."""
    root = tmp_path_factory.mktemp("run")
    graph_dir = root / "graphs"
    save_graph(build_graph(FIXTURE_ID, vtk_path, csv_path, fem_path, deck=deck), graph_dir)

    split = Split(name="tiny", train=[FIXTURE_ID], val=[FIXTURE_ID],
                  test=[FIXTURE_ID], seed=0)
    split_path = split.save(root / "split.json")
    return {"root": root, "graph_dir": str(graph_dir), "split_path": str(split_path)}


def tiny_config(workspace, **overrides) -> TrainConfig:
    """Small enough to run on a CPU in seconds."""
    defaults = dict(
        run_name="test",
        graph_dir=workspace["graph_dir"],
        split_path=workspace["split_path"],
        out_dir=str(workspace["root"] / "outputs"),
        hidden_dim=8,
        num_blocks=2,
        batch_size=1,
        max_epochs=3,
        warmup_epochs=1,
        patience=1000,
        num_workers=0,
        device="cpu",
        amp=False,
        in_memory=True,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


class TestMetrics:
    def test_r2_of_a_perfect_prediction(self):
        y = torch.randn(100, 1)
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_r2_of_predicting_the_mean(self):
        """The trivial baseline. If a trained model cannot beat this, it has learned
        nothing -- so the number it must clear is zero, not some arbitrary threshold."""
        y = torch.randn(200, 1)
        assert r2_score(torch.full_like(y, float(y.mean())), y) == pytest.approx(0.0, abs=1e-6)

    def test_r2_can_go_negative(self):
        y = torch.randn(100, 1)
        assert r2_score(-y, y) < 0

    def test_r2_of_a_constant_target_is_nan(self):
        # Zero variance to explain; guarded so it returns NaN rather than dividing by 0.
        y = torch.full((50, 1), 3.0)
        assert math_isnan(r2_score(y * 0.9, y))


def math_isnan(value: float) -> bool:
    return value != value


class TestLoss:
    def test_stress_only_ignores_extra_channels(self):
        config = TrainConfig(use_aux_displacement=False)
        pred = torch.randn(20, 1)
        target = torch.randn(20, 1)
        expected = torch.nn.functional.mse_loss(pred[:, 0], target[:, 0])
        assert compute_loss(pred, target, config) == pytest.approx(float(expected))

    def test_aux_term_is_added_and_weighted(self):
        config = TrainConfig(use_aux_displacement=True, aux_weight=0.2)
        pred, target = torch.randn(20, 4), torch.randn(20, 4)
        stress_only = torch.nn.functional.mse_loss(pred[:, 0], target[:, 0])
        total = compute_loss(pred, target, config)
        assert total > stress_only

    def test_aux_weight_of_zero_matches_stress_only(self):
        """Guards the claim that the auxiliary head cannot silently take over the
        objective: at weight zero the loss must be identical."""
        pred, target = torch.randn(20, 4), torch.randn(20, 4)
        weighted = compute_loss(pred, target, TrainConfig(use_aux_displacement=True,
                                                          aux_weight=0.0))
        stress = torch.nn.functional.mse_loss(pred[:, 0], target[:, 0])
        assert weighted == pytest.approx(float(stress))


class TestTrainingRun:
    def test_completes_and_writes_its_outputs(self, workspace):
        result = train(tiny_config(workspace, run_name="basic"))
        out = Path(workspace["root"]) / "outputs" / "basic"

        assert len(result.history) == 3
        assert result.stopped_because.startswith("reached max_epochs")
        for name in ("history.csv", "checkpoint.pt", "best_model.pt",
                     "result.json", "loss_curve_basic.png"):
            assert (out / name).is_file(), f"missing {name}"

    def test_history_is_written_every_epoch(self, workspace):
        """Not buffered on purpose: a session killed at the wall clock must still leave
        a complete record of what happened up to that point."""
        train(tiny_config(workspace, run_name="hist"))
        rows = (Path(workspace["root"]) / "outputs" / "hist" / "history.csv").read_text()
        assert rows.count("\n") == 4        # header + 3 epochs

    def test_checkpoint_carries_the_config_and_normalisation(self, workspace):
        """A checkpoint alone must be enough to reproduce inference. Without the
        normalisation statistics the weights are meaningless."""
        train(tiny_config(workspace, run_name="ckpt"))
        best = torch.load(
            Path(workspace["root"]) / "outputs" / "ckpt" / "best_model.pt",
            weights_only=False,
        )
        assert "normalization_stats" in best
        assert best["config"]["hidden_dim"] == 8
        assert "model" in best

    def test_metrics_are_reported_in_mpa_not_log_space(self, workspace):
        """R^2 computed on a log target flatters the model, because the log compresses
        exactly the large errors that matter. MAE in MPa is the check: in log space it
        would be a number near 1, in MPa it is hundreds."""
        result = train(tiny_config(workspace, run_name="units"))
        assert result.history[-1].val_mae_mpa > 10.0

    def test_deterministic_for_a_fixed_seed(self, workspace):
        a = train(tiny_config(workspace, run_name="seed_a", seed=7))
        b = train(tiny_config(workspace, run_name="seed_b", seed=7))
        assert a.history[0].train_loss == pytest.approx(b.history[0].train_loss, rel=1e-4)


class TestStopping:
    def test_early_stopping_fires(self, workspace):
        """With patience 0, any epoch that fails to improve must end the run --
        the mechanism that stops a long job training into the overfit regime."""
        result = train(tiny_config(workspace, run_name="early",
                                   max_epochs=50, patience=0))
        assert "early stopping" in result.stopped_because
        assert len(result.history) < 50

    def test_wall_clock_budget_fires(self, workspace):
        """The Kaggle-session guard. Stopping cleanly leaves a resumable checkpoint;
        being killed by the platform does not."""
        result = train(tiny_config(workspace, run_name="clock",
                                   max_epochs=1000, max_hours=1e-9))
        assert "wall-clock budget" in result.stopped_because
        assert "resume" in result.stopped_because


class TestResume:
    def test_a_second_call_continues_rather_than_restarting(self, workspace):
        """The property the whole checkpoint design exists for: 3,000 epochs spans
        several Kaggle sessions, so a rerun must pick up where the last one stopped."""
        first = train(tiny_config(workspace, run_name="resume", max_epochs=2))
        assert len(first.history) == 2

        second = train(tiny_config(workspace, run_name="resume", max_epochs=5))
        assert len(second.history) == 5, "restarted from scratch instead of resuming"
        assert [r.epoch for r in second.history] == [0, 1, 2, 3, 4]

    def test_resumed_history_keeps_the_original_epochs(self, workspace):
        train(tiny_config(workspace, run_name="resume2", max_epochs=2))
        before = (Path(workspace["root"]) / "outputs" / "resume2" / "history.csv").read_text()
        first_row = before.splitlines()[1]

        train(tiny_config(workspace, run_name="resume2", max_epochs=4))
        after = (Path(workspace["root"]) / "outputs" / "resume2" / "history.csv").read_text()
        assert after.splitlines()[1] == first_row, "earlier epochs were overwritten"


class TestPlot:
    def test_loss_curve_is_written(self, tmp_path):
        history = [
            EpochRecord(epoch=i, train_loss=1.0 / (i + 1), val_loss=1.2 / (i + 1),
                        val_r2_mpa=0.1 * i, val_mae_mpa=100.0 - i, lr=1e-3, seconds=1.0)
            for i in range(10)
        ]
        path = tmp_path / "curve.png"
        plot_loss_curve(history, path, "test")
        assert path.is_file() and path.stat().st_size > 1000
