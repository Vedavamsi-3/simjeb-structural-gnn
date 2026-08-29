"""Tests for the MeshGraphNet.

The interesting properties of a message-passing network are not its output shapes but
its *information flow*: how far news travels per block, and whether the residual
structure actually lets gradients reach the first layer. Those are what these test.
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from src.models.gnn import (
    MeshGraphNet,
    ProcessorBlock,
    estimate_activation_memory_gb,
    make_mlp,
)


def chain_graph(n: int, node_dim: int = 8, edge_dim: int = 4) -> Data:
    """A line: 0 -- 1 -- 2 -- ... -- n-1, edges in both directions.

    A chain makes hop distance exactly equal to index distance, so the receptive
    field of the network can be read off directly.
    """
    src = torch.arange(n - 1)
    dst = src + 1
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    return Data(
        x=torch.randn(n, node_dim),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.shape[1], edge_dim),
    )


class TestMLP:
    def test_output_shape(self):
        mlp = make_mlp(5, 32, 7)
        assert mlp(torch.randn(10, 5)).shape == (10, 7)

    def test_layer_norm_present_by_default(self):
        assert isinstance(make_mlp(4, 8, 4)[-1], torch.nn.LayerNorm)

    def test_decoder_style_has_no_layer_norm(self):
        # The decoder emits a regression value, not a latent state -- normalising it
        # would force the output to zero mean and unit variance per sample.
        assert isinstance(make_mlp(4, 8, 1, layer_norm=False)[-1], torch.nn.Linear)


class TestProcessorBlock:
    def test_shapes_preserved(self):
        block = ProcessorBlock(16)
        g = chain_graph(6)
        h, e = torch.randn(6, 16), torch.randn(g.edge_index.shape[1], 16)
        h2, e2 = block(h, e, g.edge_index)
        assert h2.shape == h.shape and e2.shape == e.shape

    def test_is_residual(self):
        """With the block's own MLPs zeroed, the state must pass through untouched.
        If it did not, the connection is not actually residual."""
        block = ProcessorBlock(16)
        for p in list(block.edge_mlp.parameters()) + list(block.node_mlp.parameters()):
            torch.nn.init.zeros_(p)
        g = chain_graph(6)
        h, e = torch.randn(6, 16), torch.randn(g.edge_index.shape[1], 16)
        h2, e2 = block(h, e, g.edge_index)
        torch.testing.assert_close(h2, h)
        torch.testing.assert_close(e2, e)

    def test_isolated_node_receives_nothing(self):
        """A node with no incoming edges must still be updated from its own state --
        and must not produce NaN from an empty aggregation."""
        block = ProcessorBlock(16)
        data = Data(
            x=torch.randn(3, 8),
            edge_index=torch.tensor([[0], [1]]),   # node 2 is isolated
            edge_attr=torch.randn(1, 4),
        )
        h, e = torch.randn(3, 16), torch.randn(1, 16)
        h2, _ = block(h, e, data.edge_index)
        assert torch.isfinite(h2).all()


class TestForward:
    def test_output_shape(self):
        model = MeshGraphNet(node_dim=8, hidden_dim=32, num_blocks=3)
        g = chain_graph(20)
        assert model.forward_data(g).shape == (20, 1)

    def test_aux_head_widens_the_output(self):
        model = MeshGraphNet(node_dim=8, hidden_dim=32, num_blocks=2, out_dim=4)
        assert model.forward_data(chain_graph(20)).shape == (20, 4)

    def test_wider_input_for_material_flag(self):
        # node_dim tracks FeatureConfig: 8 base, 11 with material or position.
        model = MeshGraphNet(node_dim=11, hidden_dim=32, num_blocks=2)
        assert model.forward_data(chain_graph(20, node_dim=11)).shape == (20, 1)

    def test_finite_output(self):
        model = MeshGraphNet(node_dim=8, hidden_dim=32, num_blocks=15)
        assert torch.isfinite(model.forward_data(chain_graph(50))).all()

    def test_deterministic(self):
        model = MeshGraphNet(node_dim=8, hidden_dim=32, num_blocks=3).eval()
        g = chain_graph(20)
        with torch.no_grad():
            torch.testing.assert_close(model.forward_data(g), model.forward_data(g))


class TestReceptiveField:
    """How far information travels per block -- the property that sets num_blocks.

    Measured with gradients: if the prediction at node 0 depends on the input at node
    k, then d(output_0)/d(x_k) is non-zero. On a chain, node k is exactly k hops away.
    """

    @staticmethod
    def influenced_nodes(num_blocks: int, chain_length: int = 12) -> set[int]:
        torch.manual_seed(0)
        model = MeshGraphNet(node_dim=8, hidden_dim=16, num_blocks=num_blocks)
        g = chain_graph(chain_length)
        g.x.requires_grad_(True)
        model.forward_data(g)[0].sum().backward()
        grad = g.x.grad.abs().sum(dim=1)
        return {int(i) for i in torch.nonzero(grad > 1e-12).flatten()}

    def test_one_block_reaches_one_hop(self):
        assert self.influenced_nodes(1) == {0, 1}

    def test_two_blocks_reach_two_hops(self):
        assert self.influenced_nodes(2) == {0, 1, 2}

    def test_five_blocks_reach_five_hops(self):
        assert self.influenced_nodes(5) == {0, 1, 2, 3, 4, 5}

    def test_information_does_not_travel_further_than_the_block_count(self):
        """The constraint that makes num_blocks a real design decision: a node cannot
        be influenced by anything more than num_blocks hops away, so the load path
        must fit inside that radius."""
        assert 7 not in self.influenced_nodes(3)


class TestGradients:
    def test_gradients_reach_the_first_encoder(self):
        """15 residual blocks is deep. If gradients vanished before reaching the node
        encoder, the input features would never be learned from."""
        model = MeshGraphNet(node_dim=8, hidden_dim=32, num_blocks=15)
        model.forward_data(chain_graph(30)).sum().backward()

        first_weight = model.node_encoder[0].weight
        assert first_weight.grad is not None
        assert float(first_weight.grad.abs().max()) > 0

    def test_every_parameter_receives_gradient(self):
        model = MeshGraphNet(node_dim=8, hidden_dim=16, num_blocks=4)
        model.forward_data(chain_graph(30)).sum().backward()
        dead = [n for n, p in model.named_parameters()
                if p.grad is None or float(p.grad.abs().max()) == 0.0]
        assert not dead, f"parameters with no gradient: {dead[:5]}"


class TestBatching:
    def test_batch_matches_individual_forward_passes(self):
        """Graphs in a batch must not influence each other. If PyG's index offsetting
        failed, messages would cross between brackets and this would diverge."""
        torch.manual_seed(0)
        model = MeshGraphNet(node_dim=8, hidden_dim=16, num_blocks=3).eval()
        a, b = chain_graph(10), chain_graph(7)

        with torch.no_grad():
            separate = torch.cat([model.forward_data(a), model.forward_data(b)])
            batched = model.forward_data(Batch.from_data_list([a, b]))

        torch.testing.assert_close(batched, separate, atol=1e-5, rtol=1e-4)


class TestCapacity:
    def test_default_configuration(self):
        """The default is 64 x 8 -- 0.25M parameters -- not the paper's 128 x 15.

        Depth is set by the length scale of stress concentration (fillet radii, a few
        mm) rather than by the load path, because the distance-to-boundary features
        already supply the global context. See MeshGraphNet's docstring.
        """
        model = MeshGraphNet(node_dim=8)
        assert model.hidden_dim == 64 and model.num_blocks == 8
        assert 0.2 < model.num_parameters / 1e6 < 0.35

    def test_paper_configuration_is_much_larger(self):
        """Pinned because the plan estimated 2-3M by hand and was wrong: the edge MLP's
        3*hidden input dominates and is easy to under-count."""
        model = MeshGraphNet(node_dim=8, hidden_dim=128, num_blocks=15)
        assert 1.5 < model.num_parameters / 1e6 < 2.5

    def test_default_needs_no_checkpointing(self):
        """The reason checkpointing defaults to off: at 64 x 8 a batch of 4 fits in
        fp32 on a 16 GB GPU, so paying 30% compute would buy nothing."""
        per_graph = estimate_activation_memory_gb(50_994, 306_838, 64, 8)
        assert per_graph < 3.0
        assert per_graph * 4 < 13.0, "a batch of 4 should fit in fp32 on a T4"
        assert MeshGraphNet(node_dim=8).use_checkpointing is False

    def test_depth_covers_the_stress_concentration_length_scale(self):
        """8 hops at ~0.95 mm mean edge length reaches ~7.6 mm, which spans the fillet
        radii where stress actually concentrates -- the justification for the depth,
        replacing the receptive-field argument that measurement disproved."""
        mean_edge_mm = 0.948
        reach_mm = MeshGraphNet(node_dim=8).num_blocks * mean_edge_mm
        assert 2.0 < reach_mm < 12.0

    def test_depth_dominates_the_parameter_count(self):
        small = MeshGraphNet(node_dim=8, hidden_dim=128, num_blocks=1).num_parameters
        large = MeshGraphNet(node_dim=8, hidden_dim=128, num_blocks=15).num_parameters
        assert large > 10 * small

    def test_memory_estimate_scales_with_depth_and_edges(self):
        base = estimate_activation_memory_gb(50_000, 300_000, 128, 15)
        assert base > 0
        assert estimate_activation_memory_gb(50_000, 300_000, 128, 30) == pytest.approx(
            2 * base
        )
        assert estimate_activation_memory_gb(50_000, 600_000, 128, 15) > base

    def test_checkpointing_is_what_makes_the_planned_config_fit(self):
        """The measured constraint: at hidden=128, blocks=15 on a real SimJEB graph,
        storing every block's activations needs ~10.6 GB for a *single* graph, which
        does not fit a 16 GB T4. Checkpointing is not an optimisation here -- it is
        what makes the configuration trainable at all."""
        n_nodes, n_edges = 50_994, 306_838       # model 148, measured
        plain = estimate_activation_memory_gb(n_nodes, n_edges, 128, 15)
        checkpointed = estimate_activation_memory_gb(n_nodes, n_edges, 128, 15,
                                                     checkpointing=True)
        assert plain > 10.0, "the constraint this guards against has changed"
        assert checkpointed < plain / 2
        assert checkpointed < 8.0, "still would not leave room for a batch on a T4"


class TestCheckpointing:
    def test_output_matches_the_uncheckpointed_model(self):
        """Recomputing activations must not change the result -- only the memory."""
        torch.manual_seed(0)
        model = MeshGraphNet(node_dim=8, hidden_dim=16, num_blocks=4)
        g = chain_graph(20)
        model.eval()
        with torch.no_grad():
            plain = model.forward_data(g)
        model.use_checkpointing = True
        model.train()
        with torch.no_grad():
            checkpointed = model.forward_data(g)
        torch.testing.assert_close(checkpointed, plain, atol=1e-5, rtol=1e-4)

    def test_gradients_match(self):
        """The point of checkpointing is identical gradients from less memory. If they
        differed, training would silently diverge from the un-checkpointed run."""
        def grads(use_ckpt: bool):
            torch.manual_seed(0)
            model = MeshGraphNet(node_dim=8, hidden_dim=16, num_blocks=4,
                                 use_checkpointing=use_ckpt)
            torch.manual_seed(1)
            g = chain_graph(20)
            model.forward_data(g).sum().backward()
            return [p.grad.clone() for p in model.parameters()]

        for a, b in zip(grads(False), grads(True)):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
