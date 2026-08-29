"""MeshGraphNet-style encode--process--decode network for surface stress prediction.

Reference architecture: Pfaff, Fortunato, Sanchez-Gonzalez & Battaglia,
*Learning Mesh-Based Simulation with Graph Networks*, ICLR 2021 -- itself built on
the Graph Network framework of Battaglia et al. (2018).

Adapted in two ways for this problem, both worth being able to state plainly:

* MeshGraphNets predicts the *next* state and rolls out over time. Linear-static FEA
  has no time dimension, so this predicts the field directly in one shot.
* MeshGraphNets carries two edge types (mesh-space and world-space) because it handles
  contact and self-collision. Nothing here collides, so there is one edge type.

Why this shape of network at all
--------------------------------
Strain is a spatial derivative of displacement, and stress follows from strain. A
spatial derivative is built from differences between neighbouring points -- which is
exactly what a message carrying ``x_j - x_i`` lets a layer compute. An architecture
without edge *features* (GCN, GraphSAGE, GAT) can record that two nodes are connected
but not how far apart or in which direction, so it cannot represent the derivative at
all. That requirement, not preference, is what selects this family.

Inputs are expected **already normalised** -- see ``src/data/normalization.py``. The
model deliberately owns no statistics, so a checkpoint plus its saved normaliser fully
determine inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int,
             n_hidden_layers: int = 1, layer_norm: bool = True) -> nn.Sequential:
    """The MLP used everywhere in the network.

    ``LayerNorm`` on the output of every MLP except the decoder's. Without it, 15
    stacked residual blocks drift in scale as depth accumulates and training becomes
    unstable; the decoder is left un-normalised because its output is a regression
    value, not a latent state.
    """
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class ProcessorBlock(nn.Module):
    """One round of message passing: update every edge, then every node.

    Each block moves information exactly one hop along the mesh, so the number of
    blocks sets how far a node can "see". The load path runs from the lug to the bolt
    holes, and a node needs to have heard from both ends to predict its stress.

    Both updates are residual. At this depth that is what makes the network trainable
    at all -- the same reason ResNet uses them.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # edge update sees: its own state, the sender's state, the receiver's state
        self.edge_mlp = make_mlp(3 * hidden_dim, hidden_dim, hidden_dim)
        # node update sees: its own state, and the sum of messages arriving at it
        self.node_mlp = make_mlp(2 * hidden_dim, hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor,
                edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]

        # Gather the endpoint states for every edge. This is where node data and edge
        # data meet: h[src] and h[dst] copy node rows out to edge-length tensors, so
        # all three can be concatenated.
        edge_input = torch.cat([e, h[src], h[dst]], dim=1)
        delta_e = self.edge_mlp(edge_input)

        # Sum the messages arriving at each node. Sum rather than mean: a node with
        # more neighbours genuinely has more material attached to it, and averaging
        # would erase that.
        aggregated = scatter(delta_e, dst, dim=0, dim_size=h.shape[0], reduce="sum")

        delta_h = self.node_mlp(torch.cat([h, aggregated], dim=1))

        return h + delta_h, e + delta_e


class MeshGraphNet(nn.Module):
    """Encode -> process -> decode.

    Parameters
    ----------
    node_dim, edge_dim
        Input widths. ``node_dim`` follows the feature flags in ``FeatureConfig``
        (8 by default, 11 with material or position, 14 with both); ``edge_dim`` is 4.
    hidden_dim
        Latent width used throughout. Governs how many distinct *local* configurations
        a node can encode, so it is the dimension to raise when the geometry is varied
        rather than when context needs to travel further.
    num_blocks
        Message-passing rounds, i.e. how many hops a node can see.
    out_dim
        1 for stress alone; 4 when the auxiliary displacement head is on, with stress
        in channel 0 so the primary objective is unchanged either way.

    Choosing the defaults
    ---------------------
    The MeshGraphNets paper uses ``hidden_dim=128, num_blocks=15``; those defaults do
    not transfer here, and the reason is worth recording.

    The usual justification for depth is that a node's receptive field should span the
    load path. Measured on a real SimJEB surface graph, the lug is **54 hops** from the
    nearest bolt hole and 81 from the farthest -- so 15 blocks reaches barely a quarter
    of it, and 54 blocks would cost 10.6 GB per graph. The paper's meshes are an order
    of magnitude coarser than these, so hop counts do not carry across.

    Spanning the load path turns out to be the wrong target anyway:

    * Stress concentration is a **local** phenomenon, set by fillet radii and thickness
      changes. At ~0.95 mm mean edge length, 8 hops reaches ~7.6 mm, which covers that
      length scale.
    * The **global** context -- where a node sits between the clamps and the load -- is
      supplied directly by the ``dist_to_clamp`` and ``dist_to_load`` node features,
      with no message passing required. Those features are therefore load-bearing, not
      conveniences.

    ``64 x 8`` also fits comfortably in memory (2.83 GB per graph, batch of 4 in fp32),
    which removes the need for gradient checkpointing entirely.

    One caution on capacity: 274 training graphs of ~51k nodes looks like 14M
    supervised points, but neighbouring nodes carry nearly identical stress, so the
    effective sample size for learning a shape-to-stress mapping is closer to the
    **274 distinct geometries**. Against that, 0.25M parameters is already generous.
    Runs A and B exist to probe this envelope rather than trust the estimate.
    """

    def __init__(self, node_dim: int, edge_dim: int = 4, hidden_dim: int = 64,
                 num_blocks: int = 8, out_dim: int = 1,
                 use_checkpointing: bool = False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        # Gradient checkpointing: discard each block's internal activations and
        # recompute them on the backward pass. Roughly 30% slower for roughly 3x less
        # memory.
        #
        # OFF by default, deliberately. At the default 64 x 8 a graph needs 2.83 GB and
        # a batch of 4 fits in fp32 on a 16 GB GPU, so there is no memory problem to
        # solve and the 30% would be pure waste. It exists for the wide-and-deep
        # configurations explored in run B, where 128 x 15 needs 10.6 GB per graph and
        # does not otherwise fit at all.
        self.use_checkpointing = use_checkpointing

        self.node_encoder = make_mlp(node_dim, hidden_dim, hidden_dim)
        self.edge_encoder = make_mlp(edge_dim, hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(ProcessorBlock(hidden_dim) for _ in range(num_blocks))
        self.decoder = make_mlp(hidden_dim, hidden_dim, out_dim, layer_norm=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                # Only during training -- inference stores nothing to begin with.
                h, e = torch.utils.checkpoint.checkpoint(
                    block, h, e, edge_index, use_reentrant=False
                )
            else:
                h, e = block(h, e, edge_index)
        return self.decoder(h)

    def forward_data(self, data) -> torch.Tensor:
        """Convenience wrapper for a PyG ``Data`` or ``Batch``."""
        return self.forward(data.x, data.edge_index, data.edge_attr)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        return (
            f"MeshGraphNet(node_dim={self.node_dim}, edge_dim={self.edge_dim}, "
            f"hidden={self.hidden_dim}, blocks={self.num_blocks}, "
            f"out={self.out_dim}) -- {self.num_parameters:,} parameters"
        )


def estimate_activation_memory_gb(n_nodes: int, n_edges: int, hidden_dim: int,
                                  num_blocks: int, bytes_per_value: int = 4,
                                  checkpointing: bool = False) -> float:
    """Rough activation memory for one training step, in GB.

    Backpropagation keeps every block's intermediate tensors alive, so memory grows
    linearly with depth. On these graphs the edge tensors dominate -- there are ~6
    times as many edges as nodes, and the edge update concatenates three hidden-width
    tensors across all of them -- which constrains batch size far more here than in a
    typical GNN.

    With ``checkpointing``, only the state passed between blocks is retained and each
    block's interior is recomputed on the backward pass. Memory then scales with the
    boundary states plus one block's working set, not with depth times working set.

    Deliberately approximate: it counts block inputs and outputs, not every internal
    MLP activation, so treat it as a lower bound. Measure on the real GPU before
    fixing a batch size.
    """
    per_block = (
        n_edges * 3 * hidden_dim      # concatenated edge input -- the dominant term
        + n_edges * hidden_dim        # edge update
        + n_nodes * 2 * hidden_dim    # concatenated node input
        + n_nodes * hidden_dim        # node update
    )
    if checkpointing:
        boundary = (n_nodes + n_edges) * hidden_dim   # h and e handed between blocks
        values = boundary * num_blocks + per_block    # + one block recomputed at a time
    else:
        values = per_block * num_blocks
    return values * bytes_per_value / 1e9
