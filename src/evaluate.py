"""Final evaluation on the held-out test set.

Opened once, at the end. Every look at the test set during development turns it into a
second validation set, and model selection slowly overfits to it -- so this runs after
all training decisions are locked.

Three things are reported, and the last two are what make the first meaningful:

**Aggregate metrics in MPa.** R^2, MAE and RMSE, computed after inverting the log
transform. R^2 on a log target flatters the model, because the log compresses exactly
the large errors that matter.

**A bootstrap interval.** With 77 test models a point estimate is a guess dressed as a
measurement. Resampling the test models gives an honest error bar for free -- no
retraining.

**A trivial baseline.** Predicting the training mean at every node should score ~0 R^2.
If it scores well, the task or the split is too easy and the headline number means
nothing. Reporting it costs one line and pre-empts the obvious question.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import FeatureConfig, SimJEBDataset, stress_inverse
from src.data.normalization import NormalizationStats
from src.data.split import Split
from src.models.gnn import MeshGraphNet


@dataclass
class ModelResult:
    """Per-model scores, so the tail of the distribution stays visible."""

    model_id: int
    r2: float
    mae_mpa: float
    rmse_mpa: float
    max_true_mpa: float
    max_pred_mpa: float
    n_nodes: int


@dataclass
class EvaluationReport:
    run_name: str
    split_name: str
    n_models: int
    r2: float
    mae_mpa: float
    rmse_mpa: float
    r2_ci95: tuple[float, float]
    baseline_r2: float
    per_model: list[ModelResult] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def load_checkpoint(path: str | Path, device: torch.device
                    ) -> tuple[MeshGraphNet, NormalizationStats, FeatureConfig, dict]:
    """Rebuild a trained model from its checkpoint alone.

    The checkpoint carries the weights, the normalisation statistics and the full
    config, because weights without statistics are meaningless -- the model expects
    normalised input and has no idea what scale its training data was on.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    config = state["config"]

    feature_config = FeatureConfig(
        load_case=config["load_case"],
        use_material=config["use_material"],
        use_position=config["use_position"],
        use_aux_displacement=config["use_aux_displacement"],
        log_stress=config["log_stress"],
    )
    model = MeshGraphNet(
        node_dim=feature_config.node_feature_dim,
        edge_dim=4,
        hidden_dim=config["hidden_dim"],
        num_blocks=config["num_blocks"],
        out_dim=feature_config.target_dim,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    stats = NormalizationStats.from_state_dict(state["normalization_stats"]).to(device)
    return model, stats, feature_config, config


@torch.no_grad()
def predict(model: MeshGraphNet, stats: NormalizationStats, data,
            log_stress: bool, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and true stress for one graph, both in MPa."""
    data = data.to(device)
    prediction = model(
        stats.node.normalize(data.x),
        data.edge_index,
        stats.edge.normalize(data.edge_attr),
    )
    stress = stats.target.denormalize(prediction)[:, 0:1]
    if log_stress:
        stress = stress_inverse(stress)
    return (stress.float().cpu().numpy().ravel(),
            data.stress_mpa.float().cpu().numpy().ravel())


def bootstrap_r2_interval(per_model: list[ModelResult], n_resamples: int = 2000,
                          seed: int = 0) -> tuple[float, float]:
    """95% interval for the pooled R^2, by resampling whole test models.

    Models are resampled, not nodes: nodes within a bracket are highly correlated, so
    resampling them would treat 4M correlated points as 4M independent ones and produce
    an interval far too narrow to be honest. The unit of independence here is the
    geometry.
    """
    rng = np.random.default_rng(seed)
    scores = np.array([m.r2 for m in per_model])
    weights = np.array([m.n_nodes for m in per_model], dtype=float)

    draws = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(scores), len(scores))
        draws.append(float(np.average(scores[idx], weights=weights[idx])))
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def evaluate(checkpoint_path: str | Path, graph_dir: str | Path,
             split_path: str | Path, run_name: str = "C",
             device: str = "cuda", out_dir: str | Path = "outputs") -> EvaluationReport:
    """Score a trained model on its split's test set."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, stats, feature_config, config = load_checkpoint(checkpoint_path, device)
    split = Split.load(split_path)

    test_set = SimJEBDataset(graph_dir, split.test, feature_config)
    train_set = SimJEBDataset(graph_dir, split.train, feature_config)

    # The trivial baseline: the mean stress over the training split, in MPa.
    train_mean = float(np.mean([
        float(train_set[i].stress_mpa.mean()) for i in range(len(train_set))
    ]))

    per_model, all_pred, all_true = [], [], []
    for i in range(len(test_set)):
        data = test_set[i]
        pred, true = predict(model, stats, data, feature_config.log_stress, device)
        per_model.append(ModelResult(
            model_id=int(data.model_id),
            r2=_r2(pred, true),
            mae_mpa=float(np.abs(pred - true).mean()),
            rmse_mpa=float(np.sqrt(((pred - true) ** 2).mean())),
            max_true_mpa=float(true.max()),
            max_pred_mpa=float(pred.max()),
            n_nodes=int(true.size),
        ))
        all_pred.append(pred)
        all_true.append(true)

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    report = EvaluationReport(
        run_name=run_name,
        split_name=split.name,
        n_models=len(per_model),
        r2=_r2(pred, true),
        mae_mpa=float(np.abs(pred - true).mean()),
        rmse_mpa=float(np.sqrt(((pred - true) ** 2).mean())),
        r2_ci95=bootstrap_r2_interval(per_model),
        baseline_r2=_r2(np.full_like(true, train_mean), true),
        per_model=sorted(per_model, key=lambda m: m.r2),
        notes={
            "metrics_in": "MPa (log transform inverted before scoring)",
            "load_case": feature_config.load_case,
            "checkpoint": str(checkpoint_path),
        },
    )
    report.save(Path(out_dir) / run_name / "evaluation.json")

    print(f"{run_name} on {split.name}: {report.n_models} models")
    print(f"  R2   {report.r2:.4f}   95% CI [{report.r2_ci95[0]:.4f}, {report.r2_ci95[1]:.4f}]")
    print(f"  MAE  {report.mae_mpa:.1f} MPa    RMSE {report.rmse_mpa:.1f} MPa")
    print(f"  trivial baseline R2: {report.baseline_r2:.4f}")
    worst = report.per_model[:3]
    print("  worst models: " + ", ".join(f"{m.model_id} (R2 {m.r2:.3f})" for m in worst))
    return report


def plot_per_model_r2(report: EvaluationReport, path: str | Path,
                      flagged: set[int] | None = None) -> Path:
    """Every test model's R^2, worst first.

    A single averaged number hides whether the model is uniformly decent or excellent
    on most brackets and useless on a few. With 77 test models every one can be shown,
    and the tail is the interesting part -- especially cross-referenced against the QA
    flags, which is what ``flagged`` marks.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flagged = flagged or set()
    scores = [m.r2 for m in report.per_model]
    colours = ["tab:orange" if m.model_id in flagged else "tab:blue"
               for m in report.per_model]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(scores)), scores, color=colours)
    ax.axhline(report.r2, color="black", linestyle="--", linewidth=1,
               label=f"pooled $R^2$ = {report.r2:.3f}")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("test model (worst first)")
    ax.set_ylabel("$R^2$")
    ax.set_title(f"{report.run_name} -- per-model accuracy on {report.split_name}")
    if flagged:
        ax.bar([], [], color="tab:orange", label="flagged by QA")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prediction_vs_truth(report: EvaluationReport, checkpoint_path: str | Path,
                             graph_dir: str | Path, split_path: str | Path,
                             path: str | Path, device: str = "cuda",
                             max_points: int = 50_000) -> Path:
    """Predicted against true stress, pooled over the test set.

    Shows *where* the model fails rather than by how much on average. Systematic
    curvature means bias; a fan opening at high stress means the singular peaks are
    the hard part -- which is the expected failure mode here, and worth confirming
    rather than assuming.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model, stats, feature_config, _ = load_checkpoint(checkpoint_path, device_t)
    split = Split.load(split_path)
    test_set = SimJEBDataset(graph_dir, split.test, feature_config)

    preds, trues = [], []
    for i in range(len(test_set)):
        p, t = predict(model, stats, test_set[i], feature_config.log_stress, device_t)
        preds.append(p)
        trues.append(t)
    pred, true = np.concatenate(preds), np.concatenate(trues)

    if pred.size > max_points:
        idx = np.random.default_rng(0).choice(pred.size, max_points, replace=False)
        pred, true = pred[idx], true[idx]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(true, pred, s=1, alpha=0.15, edgecolors="none")
    limit = float(max(true.max(), pred.max()))
    ax.plot([0, limit], [0, limit], "k--", linewidth=1, label="perfect")
    ax.axvline(880, color="tab:red", linestyle=":", linewidth=1,
               label="Ti-6Al-4V yield (880 MPa)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true von Mises (MPa)")
    ax.set_ylabel("predicted von Mises (MPa)")
    ax.set_title(f"{report.run_name}: $R^2$ = {report.r2:.3f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
