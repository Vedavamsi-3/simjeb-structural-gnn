"""Tests for dataset assembly, archive access, and final evaluation.

These cover the two stages that only ever run at full scale -- building 381 graphs and
scoring the test set -- so the properties worth asserting are the ones that would
otherwise only reveal themselves after an hour of Kaggle time: that a failed model is
recorded rather than dropped, that a rerun resumes, and that the reported number is in
MPa rather than log space.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.build_graph import build_graph, save_graph
from src.data.fetch import FILES, SimJEBSource
from src.data.make_dataset import BuildReport, dataset_summary, make_dataset
from src.data.split import Split
from src.evaluate import (
    _r2,
    bootstrap_r2_interval,
    evaluate,
    load_checkpoint,
    ModelResult,
    plot_per_model_r2,
    plot_prediction_vs_truth,
)
from src.train import TrainConfig, train
from tests.conftest import FIXTURE_ID


@pytest.fixture(scope="module")
def archive_dir(vtk_path, csv_path, fem_path, tmp_path_factory):
    """A miniature stand-in for the real distribution: the fixture's three files,
    packed into two zips so the split-archive handling is exercised."""
    root = tmp_path_factory.mktemp("archives")
    with zipfile.ZipFile(root / "meshes.zip", "w") as z:
        z.write(vtk_path, f"{FIXTURE_ID}.vtk")
    with zipfile.ZipFile(root / "inputs.zip", "w") as z:
        z.write(fem_path, f"{FIXTURE_ID}.fem")
        z.write(csv_path, f"{FIXTURE_ID}field.csv")
    return root


class TestFetch:
    def test_published_sizes_are_recorded(self):
        """Sizes are checked after download so a reset connection produces an error
        rather than a truncated archive that fails later, somewhere confusing."""
        for key, (file_id, size, name) in FILES.items():
            assert file_id > 0 and size > 0 and name

    def test_indexes_across_multiple_archives(self, archive_dir):
        """The .fem decks and result CSVs are split across two zips each, with no
        documented rule for which model lands where -- so the index is built by
        looking, not by assuming."""
        source = SimJEBSource.open(archive_dir)
        assert source.model_ids() == [FIXTURE_ID]

    def test_only_lists_models_with_all_three_files(self, tmp_path, vtk_path):
        # A mesh with no results and no deck cannot be built, so it must not appear.
        with zipfile.ZipFile(tmp_path / "partial.zip", "w") as z:
            z.write(vtk_path, "999.vtk")
        assert SimJEBSource.open(tmp_path).model_ids() == []

    def test_reads_a_member_without_extracting(self, archive_dir):
        data = SimJEBSource.open(archive_dir).read(f"{FIXTURE_ID}.fem")
        assert data.startswith(b"$$")

    def test_missing_member_raises(self, archive_dir):
        with pytest.raises(KeyError, match="not found in any archive"):
            SimJEBSource.open(archive_dir).read("12345.fem")

    def test_no_archives_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no SimJEB archives"):
            SimJEBSource.open(tmp_path)

    def test_extract_model_writes_all_three(self, archive_dir, tmp_path):
        paths = SimJEBSource.open(archive_dir).extract_model(FIXTURE_ID, tmp_path)
        assert set(paths) == {"vtk", "fem", "csv"}
        assert all(p.is_file() and p.stat().st_size > 0 for p in paths.values())


class TestMakeDataset:
    def test_builds_and_reports(self, archive_dir, tmp_path):
        report = make_dataset(archive_dir, tmp_path / "graphs",
                              scratch_dir=tmp_path / "scratch")
        assert report.built == [FIXTURE_ID]
        assert not report.failed
        assert (tmp_path / "graphs" / f"{FIXTURE_ID}.pt").is_file()
        assert report.stats[str(FIXTURE_ID)]["n_surface_nodes"] == 50_994

    def test_rerun_resumes_rather_than_rebuilding(self, archive_dir, tmp_path):
        """20 minutes of work should not be discarded because a session died at
        model 300."""
        out = tmp_path / "graphs"
        make_dataset(archive_dir, out, scratch_dir=tmp_path / "s1")
        again = make_dataset(archive_dir, out, scratch_dir=tmp_path / "s2")
        assert again.built == []
        assert again.skipped == [FIXTURE_ID]

    def test_overwrite_forces_a_rebuild(self, archive_dir, tmp_path):
        out = tmp_path / "graphs"
        make_dataset(archive_dir, out, scratch_dir=tmp_path / "s1")
        forced = make_dataset(archive_dir, out, scratch_dir=tmp_path / "s2",
                              overwrite=True)
        assert forced.built == [FIXTURE_ID]

    def test_a_bad_model_is_recorded_not_dropped(self, archive_dir, tmp_path):
        """The property that matters most here. A pipeline that silently skips
        failures produces a dataset that is not what you think you trained on."""
        report = make_dataset(archive_dir, tmp_path / "graphs",
                              scratch_dir=tmp_path / "scratch",
                              model_ids=[FIXTURE_ID, 99_999])
        assert report.built == [FIXTURE_ID]
        assert "99999" in report.failed
        assert report.failed["99999"]

    def test_scratch_is_cleaned_up(self, archive_dir, tmp_path):
        """Peak disk must stay at one model. Leaving extracted files behind would
        exhaust Kaggle's 20 GB working directory partway through."""
        scratch = tmp_path / "scratch"
        make_dataset(archive_dir, tmp_path / "graphs", scratch_dir=scratch)
        assert not any(scratch.iterdir())

    def test_report_serialises(self, tmp_path):
        report = BuildReport(built=[1, 2], failed={"3": "bad mesh"})
        loaded = json.loads(report.save(tmp_path / "r.json").read_text())
        assert loaded["built"] == [1, 2]
        assert loaded["failed"]["3"] == "bad mesh"

    def test_summary_over_the_cache(self, archive_dir, tmp_path):
        out = tmp_path / "graphs"
        make_dataset(archive_dir, out, scratch_dir=tmp_path / "scratch")
        summary = dataset_summary(out)
        assert summary["n_models"] == 1
        assert summary["surface_nodes"]["total"] == 50_994
        assert summary["disk_gb"] > 0

    def test_summary_of_an_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no cached graphs"):
            dataset_summary(tmp_path)


class TestBootstrap:
    def test_interval_brackets_the_estimate(self):
        models = [ModelResult(i, 0.8 + 0.02 * (i % 5), 10, 12, 900, 880, 1000)
                  for i in range(40)]
        low, high = bootstrap_r2_interval(models, n_resamples=500)
        assert low < np.mean([m.r2 for m in models]) < high

    def test_more_variance_widens_the_interval(self):
        tight = [ModelResult(i, 0.80, 10, 12, 900, 880, 1000) for i in range(40)]
        loose = [ModelResult(i, 0.80 + 0.3 * ((-1) ** i), 10, 12, 900, 880, 1000)
                 for i in range(40)]
        t = bootstrap_r2_interval(tight, n_resamples=500)
        l = bootstrap_r2_interval(loose, n_resamples=500)
        assert (l[1] - l[0]) > (t[1] - t[0])

    def test_resamples_models_not_nodes(self):
        """Nodes within a bracket are highly correlated, so resampling them would
        treat millions of dependent points as independent and produce an interval far
        too narrow to be honest. The unit of independence is the geometry."""
        models = [ModelResult(i, 0.5 + 0.4 * (i % 2), 10, 12, 900, 880, 50_000)
                  for i in range(10)]
        low, high = bootstrap_r2_interval(models, n_resamples=500)
        assert high - low > 0.1, "interval implausibly tight for 10 varied models"

    def test_deterministic_for_a_seed(self):
        models = [ModelResult(i, 0.7 + 0.01 * i, 10, 12, 900, 880, 1000)
                  for i in range(20)]
        assert bootstrap_r2_interval(models, 300, seed=1) == \
               bootstrap_r2_interval(models, 300, seed=1)


class TestEvaluation:
    @pytest.fixture(scope="class")
    def trained(self, vtk_path, csv_path, fem_path, deck, tmp_path_factory):
        root = tmp_path_factory.mktemp("eval")
        graph_dir = root / "graphs"
        save_graph(build_graph(FIXTURE_ID, vtk_path, csv_path, fem_path, deck=deck),
                   graph_dir)
        split = Split(name="tiny", train=[FIXTURE_ID], val=[FIXTURE_ID],
                      test=[FIXTURE_ID], seed=0)
        split_path = split.save(root / "split.json")

        train(TrainConfig(
            run_name="eval", graph_dir=str(graph_dir), split_path=str(split_path),
            out_dir=str(root / "outputs"), hidden_dim=8, num_blocks=2, batch_size=1,
            max_epochs=2, warmup_epochs=1, num_workers=0, device="cpu", amp=False,
        ))
        return {
            "root": root,
            "graph_dir": str(graph_dir),
            "split_path": str(split_path),
            "checkpoint": root / "outputs" / "eval" / "best_model.pt",
        }

    def test_checkpoint_alone_rebuilds_the_model(self, trained):
        """Weights without the normalisation statistics are meaningless -- the model
        expects normalised input and has no idea what scale it was trained on."""
        model, stats, feature_config, config = load_checkpoint(
            trained["checkpoint"], torch.device("cpu")
        )
        assert model.hidden_dim == 8

        # 8 base features + 3 material (on) + 0 position (off). Derived from the
        # restored config rather than hard-coded, so a feature-flag change shows up as
        # a real mismatch rather than a stale constant in a test.
        expected = 8 + (3 if config["use_material"] else 0) \
                     + (3 if config["use_position"] else 0)
        assert feature_config.node_feature_dim == expected == 11
        assert stats.node.mean.numel() == expected

        # The auxiliary head is on, so the model emits stress plus three displacement
        # channels -- while the primary objective stays in channel 0.
        assert model.out_dim == 4 if config["use_aux_displacement"] else 1

    def test_report_is_in_mpa(self, trained):
        """MAE in log space would be a number near 1; in MPa it is hundreds. This is
        the check that the transform was actually inverted before scoring."""
        report = evaluate(trained["checkpoint"], trained["graph_dir"],
                          trained["split_path"], run_name="eval", device="cpu",
                          out_dir=str(trained["root"] / "outputs"))
        assert report.mae_mpa > 10.0
        assert report.notes["metrics_in"].startswith("MPa")

    def test_baseline_is_reported(self, trained):
        """Predicting the training mean should score ~0. If it scored well, the task
        or the split would be too easy and the headline number would mean nothing."""
        report = evaluate(trained["checkpoint"], trained["graph_dir"],
                          trained["split_path"], run_name="eval", device="cpu",
                          out_dir=str(trained["root"] / "outputs"))
        assert report.baseline_r2 < 0.2

    def test_per_model_results_are_worst_first(self, trained):
        report = evaluate(trained["checkpoint"], trained["graph_dir"],
                          trained["split_path"], run_name="eval", device="cpu",
                          out_dir=str(trained["root"] / "outputs"))
        scores = [m.r2 for m in report.per_model]
        assert scores == sorted(scores)

    def test_report_is_saved(self, trained):
        evaluate(trained["checkpoint"], trained["graph_dir"], trained["split_path"],
                 run_name="eval", device="cpu",
                 out_dir=str(trained["root"] / "outputs"))
        saved = trained["root"] / "outputs" / "eval" / "evaluation.json"
        assert json.loads(saved.read_text())["notes"]["load_case"] == "ver"

    def test_figures_are_written(self, trained, tmp_path):
        report = evaluate(trained["checkpoint"], trained["graph_dir"],
                          trained["split_path"], run_name="eval", device="cpu",
                          out_dir=str(trained["root"] / "outputs"))
        a = plot_per_model_r2(report, tmp_path / "per_model.png", flagged={FIXTURE_ID})
        b = plot_prediction_vs_truth(report, trained["checkpoint"],
                                     trained["graph_dir"], trained["split_path"],
                                     tmp_path / "scatter.png", device="cpu")
        assert a.stat().st_size > 1000 and b.stat().st_size > 1000


class TestR2Helper:
    def test_perfect(self):
        y = np.random.RandomState(0).randn(100)
        assert _r2(y, y) == pytest.approx(1.0)

    def test_mean_predictor_scores_zero(self):
        y = np.random.RandomState(0).randn(200)
        assert _r2(np.full_like(y, y.mean()), y) == pytest.approx(0.0, abs=1e-9)
