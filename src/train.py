"""Training loop for the SimJEB stress surrogate.

Designed around three constraints that come from the problem rather than from taste:

**Sessions are shorter than the run.** At roughly 65-130 s per epoch on a T4, a
3,000-epoch budget is 55-110 hours -- several Kaggle sessions, not one. So the loop
stops cleanly on a wall-clock budget, writes a full checkpoint every improvement, and
resumes exactly where it left off. A run that dies at hour 11 with nothing to show for
it is the failure mode this exists to prevent.

**The dataset is small in the way that matters.** 274 training graphs of ~51k nodes
looks like 14M supervised points, but nodes within a graph are highly correlated, so
the effective sample size is closer to the 274 distinct geometries. Regularisation and
early stopping are load-bearing, not decoration.

**The target has a heavy tail.** Peak von Mises runs to ~15,000 MPa against a ~880 MPa
yield, because a linear-elastic solve reports unbounded stress at sharp corners. The
model trains on ``log1p`` stress for that reason -- but every reported metric is
inverted back to MPa first, because R^2 on a log target flatters the model by
compressing exactly the large errors that matter.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.data.dataset import (
    FeatureConfig,
    SimJEBDataset,
    stress_inverse,
)
from src.data.normalization import NormalizationStats, compute_normalization_stats
from src.data.split import Split
from src.models.gnn import MeshGraphNet


@dataclass
class TrainConfig:
    """Everything that defines a run. Serialised into the checkpoint so a result can
    always be traced back to the settings that produced it."""

    run_name: str = "C"

    # --- data -----------------------------------------------------------------
    graph_dir: str = "data/graphs"
    split_path: str = "splits/grouped_split_v1.json"
    load_case: str = "ver"

    # --- features -------------------------------------------------------------
    # Material is on: constant across all 381 models, so it cannot help, but it costs
    # three dead columns and means the pipeline extends unchanged to a multi-material
    # dataset. Position is off: the brackets share a frame so absolute coordinates are
    # meaningful here, but with only ~274 training shapes they are also an easy thing
    # for the network to memorise instead of learning mechanics. The distance-to-clamp
    # and distance-to-load features carry the useful part of position in a form that
    # transfers to an unseen design.
    use_material: bool = True
    use_position: bool = False

    # Displacement is supervised alongside stress and discarded at inference. Stress is
    # the gradient-derived, harder quantity; displacement is smooth and easy, and
    # predicting it forces the shared trunk to learn the deformation mechanics stress
    # depends on. Weighted down deliberately -- it is a teaching aid, not a competing
    # objective.
    use_aux_displacement: bool = True
    aux_weight: float = 0.2
    log_stress: bool = True

    # Huber transition point on the normalised log-stress target. None = plain MSE.
    #
    # The evidence for this is specific to c1's failure: RMSE (174.6) came out at more
    # than double MAE (84.0), and the worst-scoring brackets were exactly those with
    # peaks above 5x yield. A small number of singular nodes dominates the gradient
    # even after log1p. Huber is quadratic near zero and linear in the tail, so a node
    # that is wildly wrong contributes a bounded gradient -- the right shape for a
    # target whose extremes are solver artefacts rather than physics worth fitting.
    huber_delta: float | None = None

    # --- model ----------------------------------------------------------------
    hidden_dim: int = 64
    num_blocks: int = 8
    dropout: float = 0.0
    use_checkpointing: bool = False

    # --- optimisation ---------------------------------------------------------
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 3000
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # --- stopping -------------------------------------------------------------
    patience: int = 200          # epochs without validation improvement
    min_delta: float = 1e-5      # smaller than this does not count as improvement

    # Early stopping compares a moving average rather than the raw validation loss.
    # c1 took the minimum of 398 noisy scores, and the minimum of noise sits below the
    # true mean by roughly the noise scale -- which is most of why its validation MAE
    # (59.5) so badly under-predicted its test MAE (84.0).
    val_smoothing: int = 10
    max_hours: float = 10.5      # leave margin inside a 12 h Kaggle session

    # --- learning-rate schedule ----------------------------------------------
    plateau_factor: float = 0.5
    plateau_patience: int = 50

    # --- housekeeping ---------------------------------------------------------
    seed: int = 0
    device: str = "cuda"
    amp: bool = True             # mixed precision; roughly halves memory and time
    num_workers: int = 2
    in_memory: bool = True
    out_dir: str = "outputs"

    @property
    def feature_config(self) -> FeatureConfig:
        return FeatureConfig(
            load_case=self.load_case,
            use_material=self.use_material,
            use_position=self.use_position,
            use_aux_displacement=self.use_aux_displacement,
            log_stress=self.log_stress,
        )


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_r2_mpa: float
    val_mae_mpa: float
    lr: float
    seconds: float


@dataclass
class TrainResult:
    config: TrainConfig
    history: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = math.inf
    stopped_because: str = ""
    total_seconds: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def r2_score(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Fraction of variance explained.

    Reported rather than raw MSE because it is scale-free and reads as a percentage to
    an engineer who does not work in ML -- which is most of the audience for a
    surrogate model.
    """
    ss_res = torch.sum((true - pred) ** 2)
    ss_tot = torch.sum((true - true.mean()) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_loss(prediction: torch.Tensor, target: torch.Tensor,
                 config: TrainConfig) -> torch.Tensor:
    """MSE on the normalised stress channel, plus an optional auxiliary term.

    With the auxiliary head on, displacement is weighted down deliberately: it is there
    to shape the shared representation, not to compete with the objective being
    measured.
    """
    if config.huber_delta:
        stress_loss = F.huber_loss(prediction[:, 0], target[:, 0],
                                   delta=config.huber_delta)
    else:
        stress_loss = F.mse_loss(prediction[:, 0], target[:, 0])

    if not config.use_aux_displacement or prediction.shape[1] == 1:
        return stress_loss

    # Displacement keeps plain MSE: it has no comparable tail, so there is nothing for
    # Huber to protect against.
    disp_loss = F.mse_loss(prediction[:, 1:], target[:, 1:])
    return stress_loss + config.aux_weight * disp_loss


def _lr_scale(epoch: int, warmup_epochs: int) -> float:
    """Linear warmup. A fresh network takes large, badly-directed steps in its first
    epochs; easing the learning rate in avoids the early divergence that otherwise
    wastes a long run."""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / warmup_epochs)


@torch.no_grad()
def evaluate_split(model: MeshGraphNet, loader: DataLoader, stats: NormalizationStats,
                   config: TrainConfig, device: torch.device) -> tuple[float, float, float]:
    """Return (loss, R^2 in MPa, MAE in MPa) over a loader.

    Metrics are computed after inverting both the normalisation and the log transform,
    so they mean what an engineer expects them to mean.
    """
    model.eval()
    total_loss, total_nodes = 0.0, 0
    preds_mpa, trues_mpa = [], []

    for batch in loader:
        batch = batch.to(device)
        x = stats.node.normalize(batch.x)
        edge_attr = stats.edge.normalize(batch.edge_attr)
        target = stats.target.normalize(batch.y)

        prediction = model(x, batch.edge_index, edge_attr)
        loss = compute_loss(prediction, target, config)

        total_loss += float(loss) * batch.num_nodes
        total_nodes += batch.num_nodes

        # Back to MPa: undo normalisation, then undo log1p.
        stress = stats.target.denormalize(prediction)[:, 0:1]
        if config.log_stress:
            stress = stress_inverse(stress)
        preds_mpa.append(stress.float().cpu())
        trues_mpa.append(batch.stress_mpa.float().cpu())

    pred = torch.cat(preds_mpa)
    true = torch.cat(trues_mpa)
    return (
        total_loss / max(total_nodes, 1),
        r2_score(pred, true),
        float((pred - true).abs().mean()),
    )


def _write_history(path: Path, history: list[EpochRecord]) -> None:
    """Rewrite the history CSV every epoch.

    Deliberately not buffered: a session that is killed at the wall clock still leaves
    a complete record of everything up to that point.
    """
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(history[0])))
        writer.writeheader()
        for record in history:
            writer.writerow(asdict(record))


def plot_loss_curve(history: list[EpochRecord], path: Path, title: str) -> None:
    """Train and validation loss on one axis -- the figure the whole run is judged by.

    The gap between the curves, and whether validation turns upward, is how over- and
    under-fitting are diagnosed. Log scale because the first epochs dwarf the rest.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r.epoch for r in history]
    fig, (ax_loss, ax_r2) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, [r.train_loss for r in history], label="train")
    ax_loss.plot(epochs, [r.val_loss for r in history], label="validation")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss (normalised)")
    ax_loss.set_title(title)
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_r2.plot(epochs, [r.val_r2_mpa for r in history], color="tab:green")
    ax_r2.set_xlabel("epoch")
    ax_r2.set_ylabel("validation $R^2$ (MPa)")
    ax_r2.set_title("held-out accuracy")
    ax_r2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(config: TrainConfig) -> TrainResult:
    """Run one training job, resuming from a checkpoint if one is present."""
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(config.out_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"

    split = Split.load(config.split_path)
    feature_config = config.feature_config

    train_set = SimJEBDataset(config.graph_dir, split.train, feature_config,
                              in_memory=config.in_memory)
    val_set = SimJEBDataset(config.graph_dir, split.val, feature_config,
                            in_memory=config.in_memory)

    # Fitted on the training split only -- statistics over the whole dataset would leak
    # information about validation and test into training.
    stats = compute_normalization_stats(train_set).to(device)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers)

    model = MeshGraphNet(
        node_dim=feature_config.node_feature_dim,
        edge_dim=4,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        out_dim=feature_config.target_dim,
        use_checkpointing=config.use_checkpointing,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config.plateau_factor, patience=config.plateau_patience
    )
    scaler = torch.amp.GradScaler(device.type, enabled=config.amp and device.type == "cuda")

    result = TrainResult(config=config)
    start_epoch = 0
    epochs_without_improvement = 0

    # --- resume ---------------------------------------------------------------
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        result.history = [EpochRecord(**r) for r in state["history"]]
        result.best_val_loss = state["best_val_loss"]
        result.best_epoch = state["best_epoch"]
        epochs_without_improvement = state["epochs_without_improvement"]
        print(f"resumed {config.run_name} from epoch {start_epoch}")

    print(f"{model.describe()}\n"
          f"train {len(train_set)} | val {len(val_set)} | device {device} | "
          f"amp {scaler.is_enabled()}")

    began = time.time()
    for epoch in range(start_epoch, config.max_epochs):
        epoch_started = time.time()

        for group in optimizer.param_groups:
            group["lr"] = group.get("initial_lr", config.lr) * _lr_scale(
                epoch, config.warmup_epochs
            ) if epoch < config.warmup_epochs else group["lr"]

        model.train()
        running_loss, running_nodes = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                prediction = model(
                    stats.node.normalize(batch.x),
                    batch.edge_index,
                    stats.edge.normalize(batch.edge_attr),
                )
                loss = compute_loss(prediction, stats.target.normalize(batch.y), config)

            scaler.scale(loss).backward()
            if config.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach().item() * batch.num_nodes
            running_nodes += batch.num_nodes

        val_loss, val_r2, val_mae = evaluate_split(model, val_loader, stats, config, device)
        scheduler.step(val_loss)

        record = EpochRecord(
            epoch=epoch,
            train_loss=running_loss / max(running_nodes, 1),
            val_loss=val_loss,
            val_r2_mpa=val_r2,
            val_mae_mpa=val_mae,
            lr=optimizer.param_groups[0]["lr"],
            seconds=time.time() - epoch_started,
        )
        result.history.append(record)
        _write_history(out_dir / "history.csv", result.history)

        # Compare the smoothed signal, not the raw epoch, so a single lucky epoch does
        # not get selected as "best".
        window = [r.val_loss for r in result.history[-config.val_smoothing:]]
        smoothed = sum(window) / len(window)
        improved = smoothed < result.best_val_loss - config.min_delta
        if improved:
            result.best_val_loss = smoothed
            result.best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "normalization_stats": stats.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_r2_mpa": val_r2,
                },
                out_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        # Full state, every epoch, so a killed session resumes rather than restarts.
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "history": [asdict(r) for r in result.history],
                "best_val_loss": result.best_val_loss,
                "best_epoch": result.best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
            },
            checkpoint_path,
        )

        if epoch % 10 == 0 or improved:
            print(f"epoch {epoch:4d} | train {record.train_loss:.5f} | "
                  f"val {val_loss:.5f} | R2 {val_r2:.4f} | MAE {val_mae:7.1f} MPa | "
                  f"{record.seconds:.0f}s{'  *' if improved else ''}")

        elapsed_hours = (time.time() - began) / 3600
        if epochs_without_improvement >= config.patience:
            result.stopped_because = f"early stopping ({config.patience} epochs without improvement)"
            break
        if elapsed_hours >= config.max_hours:
            result.stopped_because = (
                f"wall-clock budget reached at epoch {epoch}; rerun to resume"
            )
            break
    else:
        result.stopped_because = f"reached max_epochs ({config.max_epochs})"

    result.total_seconds = time.time() - began
    plot_loss_curve(result.history, out_dir / f"loss_curve_{config.run_name}.png",
                    f"run {config.run_name}")
    (out_dir / "result.json").write_text(json.dumps(
        {
            "config": asdict(config),
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "stopped_because": result.stopped_because,
            "epochs_completed": len(result.history),
            "total_hours": round(result.total_seconds / 3600, 2),
        },
        indent=2,
    ))
    print(f"\n{config.run_name}: {result.stopped_because}\n"
          f"best epoch {result.best_epoch} | val loss {result.best_val_loss:.5f}")
    return result
