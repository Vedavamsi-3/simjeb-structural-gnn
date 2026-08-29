"""Per-channel standardisation, fitted on the training split only.

Node features, edge features and targets span very different magnitudes -- a distance
in millimetres next to a one-hot flag next to a log-stress value. Without scaling, the
largest channel dominates the gradient and training is unstable.

Two rules this module exists to enforce:

**Fit on train only.** Statistics computed over the whole dataset leak information
about the validation and test splits into training. The effect is small but it is
real, and it is the kind of thing a reviewer checks first.

**Never divide by a near-zero standard deviation.** A constant column -- the material
features, or a one-hot that happens to be all zeros in a small split -- would produce
infinities and take the whole run down. Standard deviations are clamped, so a constant
column simply passes through roughly unchanged rather than exploding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Below this, a channel is treated as constant rather than scaled up enormously.
STD_FLOOR = 1e-6


@dataclass
class Normalizer:
    """Mean and standard deviation for one tensor's channels."""

    mean: torch.Tensor
    std: torch.Tensor

    # Statistics follow the tensor's dtype as well as its device. Fitting accumulates
    # in float64 for numerical stability, but the model's weights are float32 (or
    # float16 under mixed precision), and silently promoting activations to float64
    # would fail inside the first Linear layer.
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device, x.dtype) + self.mean.to(x.device, x.dtype)

    def to(self, device) -> "Normalizer":
        return Normalizer(self.mean.to(device), self.std.to(device))

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Normalizer":
        return cls(state["mean"], state["std"])

    @classmethod
    def from_accumulator(cls, count: int, total: torch.Tensor,
                         total_sq: torch.Tensor) -> "Normalizer":
        """Build from streaming sums, so the whole split never has to be in memory."""
        if count == 0:
            raise ValueError("cannot fit a normalizer on zero samples")
        mean = total / count
        var = (total_sq / count) - mean.pow(2)
        # Floating-point cancellation can push a constant channel slightly negative.
        std = var.clamp_min(0.0).sqrt().clamp_min(STD_FLOOR)
        # Accumulated in float64, stored in float32 to match the model's weights.
        return cls(mean.float(), std.float())


@dataclass
class NormalizationStats:
    """The three normalizers a model needs, kept together so they travel as a set."""

    node: Normalizer
    edge: Normalizer
    target: Normalizer

    def to(self, device) -> "NormalizationStats":
        return NormalizationStats(
            self.node.to(device), self.edge.to(device), self.target.to(device)
        )

    def state_dict(self) -> dict:
        return {
            "node": self.node.state_dict(),
            "edge": self.edge.state_dict(),
            "target": self.target.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "NormalizationStats":
        return cls(
            node=Normalizer.from_state_dict(state["node"]),
            edge=Normalizer.from_state_dict(state["edge"]),
            target=Normalizer.from_state_dict(state["target"]),
        )


def compute_normalization_stats(dataset) -> NormalizationStats:
    """Fit node, edge and target statistics over a dataset, in one streaming pass.

    Accumulates sums and sums-of-squares rather than concatenating every sample: a
    305-model split holds tens of millions of nodes, and materialising all of them to
    call ``.mean()`` would need far more memory than the machine has.
    """
    counts = {"x": 0, "edge_attr": 0, "y": 0}
    totals: dict[str, torch.Tensor] = {}
    totals_sq: dict[str, torch.Tensor] = {}

    for data in dataset:
        for key in ("x", "edge_attr", "y"):
            value = getattr(data, key).double()
            if key not in totals:
                totals[key] = torch.zeros(value.shape[1], dtype=torch.float64)
                totals_sq[key] = torch.zeros(value.shape[1], dtype=torch.float64)
            counts[key] += value.shape[0]
            totals[key] += value.sum(dim=0)
            totals_sq[key] += value.pow(2).sum(dim=0)

    if not totals:
        raise ValueError("dataset is empty; cannot fit normalization statistics")

    return NormalizationStats(
        node=Normalizer.from_accumulator(counts["x"], totals["x"], totals_sq["x"]),
        edge=Normalizer.from_accumulator(
            counts["edge_attr"], totals["edge_attr"], totals_sq["edge_attr"]
        ),
        target=Normalizer.from_accumulator(counts["y"], totals["y"], totals_sq["y"]),
    )
