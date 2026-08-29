"""PyTorch Geometric dataset over the cached SimJEB graphs.

One cached ``.pt`` per model holds geometry once and the targets for all four load
cases. This module turns that into training samples, doing at load time the work that
was deliberately left out of the cache:

* mirroring the undirected edges into both directions,
* deriving ``edge_attr`` from ``pos`` and ``edge_index``,
* selecting the configured load case,
* assembling node features according to the feature flags.

Keeping these out of the cache is what holds the dataset at ~2.4 GB rather than
several times that, and it means a feature-set change costs a config edit rather than
re-running preprocessing over 381 meshes.

Feature layout
--------------
``x`` is assembled in a fixed order so a checkpoint's input dimension is meaningful::

    node type one-hot            3   normal / fixed / loaded
    surface normal               3
    distance to nearest clamp    1
    distance to nearest load     1
    ------------------------------
    base                         8
    material (E, nu, rho)       +3   if use_material
    absolute position           +3   if use_position

Material is constant across all 381 SimJEB models, so it carries no information and is
off by default -- it exists for the ablation, and so the pipeline extends unchanged to
a multi-material dataset. Absolute position is off by default too: the brackets share a
common frame, so it is meaningful, but with ~305 training shapes it is also an easy
thing for the model to memorise instead of learning mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from src.data.build_graph import LOAD_CASES, TARGET_CHANNELS

# Index of von Mises within TARGET_CHANNELS == ("xdisp", "ydisp", "zdisp", "stress").
STRESS_CHANNEL = TARGET_CHANNELS.index("stress")
DISP_CHANNELS = slice(0, 3)

BASE_FEATURE_DIM = 8


@dataclass(frozen=True)
class FeatureConfig:
    """What goes into ``x``, and how the targets are transformed."""

    load_case: str = "ver"
    use_material: bool = False       # ablation C'
    use_position: bool = False
    use_aux_displacement: bool = False  # ablation C''
    log_stress: bool = True

    @property
    def node_feature_dim(self) -> int:
        dim = BASE_FEATURE_DIM
        if self.use_material:
            dim += 3
        if self.use_position:
            dim += 3
        return dim

    @property
    def load_case_index(self) -> int:
        if self.load_case not in LOAD_CASES:
            raise ValueError(
                f"unknown load case {self.load_case!r}; expected one of {LOAD_CASES}"
            )
        return LOAD_CASES.index(self.load_case)

    @property
    def target_dim(self) -> int:
        return 4 if self.use_aux_displacement else 1


def stress_forward(stress: torch.Tensor) -> torch.Tensor:
    """MPa -> the space the model is trained in.

    Peak von Mises across SimJEB spans roughly 300 to 15,000 MPa, far above the
    ~880 MPa yield of Ti-6Al-4V, because a linear-elastic solve reports unbounded
    stress at sharp re-entrant corners. Those singular values are numerical artefacts,
    but under a plain z-score and MSE they would supply most of the gradient and the
    model would spend its capacity fitting corners.

    ``log1p`` compresses that tail without discarding it, and equalises *relative*
    error across the range -- which matches how the error is actually judged: 20% off
    matters the same at 200 MPa and at 2,000 MPa.
    """
    return torch.log1p(stress)


def stress_inverse(value: torch.Tensor) -> torch.Tensor:
    """Back to MPa. Always invert before scoring -- R^2 on a log target flatters the
    model, because the log compresses exactly the large errors that matter."""
    return torch.expm1(value)


def edge_attributes(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Relative position vector and its length, per directed edge.

    A bare adjacency says only "connected"; the relative-position vector carries
    direction and distance, which is what lets message passing approximate a spatial
    derivative -- and strain is a spatial derivative of displacement. This is the
    MeshGraphNets edge formulation.
    """
    delta = pos[edge_index[1]] - pos[edge_index[0]]
    length = delta.norm(dim=1, keepdim=True)
    return torch.cat([delta, length], dim=1)


def to_bidirectional(edge_index_undirected: torch.Tensor) -> torch.Tensor:
    """``(E, 2)`` undirected -> ``(2, 2E)`` directed, both ways.

    Messages must flow in both directions; the cache stores each edge once purely to
    halve the file size.
    """
    e = edge_index_undirected.long().t()          # (2, E)
    return torch.cat([e, e.flip(0)], dim=1)       # (2, 2E)


def build_data(cached: dict, config: FeatureConfig) -> Data:
    """Assemble one training sample from a cached graph."""
    meta = cached["meta"]
    pos = cached["pos"]
    n = pos.shape[0]

    edge_index = to_bidirectional(cached["edge_index_undirected"])

    node_type = cached["node_type"].long()
    one_hot = torch.zeros(n, 3, dtype=torch.float32)
    one_hot[torch.arange(n), node_type] = 1.0

    features = [
        one_hot,
        cached["normals"],
        cached["dist_fixed"].unsqueeze(1),
        cached["dist_loaded"].unsqueeze(1),
    ]

    if config.use_material:
        # Constant per model -- and constant across the whole dataset. Broadcast to
        # every node so the tensor shape is uniform; the normaliser clamps its
        # near-zero standard deviation so it simply passes through unchanged.
        mat = meta["material"]
        features.append(
            torch.tensor([mat["E"], mat["nu"], mat["rho"]], dtype=torch.float32)
            .repeat(n, 1)
        )

    if config.use_position:
        features.append(pos)

    x = torch.cat(features, dim=1)

    case = config.load_case_index
    stress = cached["y"][:, case, STRESS_CHANNEL]
    if config.log_stress:
        stress = stress_forward(stress)

    if config.use_aux_displacement:
        y = torch.cat([stress.unsqueeze(1), cached["y"][:, case, DISP_CHANNELS]], dim=1)
    else:
        y = stress.unsqueeze(1)

    data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
    data.edge_attr = edge_attributes(pos, edge_index)
    data.model_id = int(meta["model_id"])
    # Kept unscaled so evaluation can report per-model error in MPa without a
    # second pass over the cache.
    data.stress_mpa = cached["y"][:, case, STRESS_CHANNEL].unsqueeze(1)
    return data


class SimJEBDataset(Dataset):
    """Cached SimJEB graphs for one split.

    Parameters
    ----------
    root
        Directory of ``<model_id>.pt`` files produced by ``make_dataset.py``.
    model_ids
        Which models this split contains. Order is preserved so a run is reproducible.
    config
        Feature and target configuration.
    in_memory
        Hold every decoded sample in RAM. At ~6 MB per cached model, a 305-model
        training split is a few GB -- fine on a Kaggle GPU instance, and it removes
        disk reads from the epoch loop. Off by default so the local machine can open
        the dataset without loading it.
    """

    def __init__(
        self,
        root: str | Path,
        model_ids: list[int],
        config: FeatureConfig | None = None,
        in_memory: bool = False,
    ):
        self.root_dir = Path(root)
        self.model_ids = list(model_ids)
        self.config = config or FeatureConfig()
        self._cache: dict[int, Data] = {} if in_memory else None

        missing = [m for m in self.model_ids if not (self.root_dir / f"{m}.pt").is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} cached graphs missing from {self.root_dir}: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        super().__init__(root=str(self.root_dir))

    def len(self) -> int:
        return len(self.model_ids)

    def get(self, idx: int) -> Data:
        model_id = self.model_ids[idx]
        if self._cache is not None and model_id in self._cache:
            return self._cache[model_id]

        cached = torch.load(self.root_dir / f"{model_id}.pt", weights_only=False)
        data = build_data(cached, self.config)

        if self._cache is not None:
            self._cache[model_id] = data
        return data

    # PyG's Dataset wants these; the cache is produced out-of-band by make_dataset.py.
    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{m}.pt" for m in self.model_ids]

    def download(self) -> None:  # pragma: no cover - nothing to fetch
        pass

    def process(self) -> None:  # pragma: no cover - built by make_dataset.py
        pass
