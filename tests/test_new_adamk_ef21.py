import unittest
from types import SimpleNamespace

import torch

from fl_framework.components.aggregators.rp_ef21 import (
    ProjectedEF21Aggregator,
)
from fl_framework.components.optimizers.rp_ef21_adam import RPEF21Adam
from fl_framework.core.hooks import Context


class ProjectedEF21AggregatorTests(unittest.TestCase):
    def _aggregator(self, seed: int = 7) -> ProjectedEF21Aggregator:
        return ProjectedEF21Aggregator(
            m=3,
            r=2,
            k_top=1,
            k_random=1,
            num_byzantine=0,
            byzantine_bound=0,
            seed=seed,
        )

    def test_tracker_uses_ef21_update(self):
        aggregator = self._aggregator()
        first = [[torch.tensor([1.0, 2.0, 3.0])] for _ in range(3)]
        aggregator.aggregate(first)

        first_rows = aggregator.last_selected_rows
        first_expected = torch.zeros(3)
        first_expected[first_rows] = torch.tensor([1.0, 2.0, 3.0])[first_rows]
        torch.testing.assert_close(aggregator._trackers[0], first_expected)

        # The optimizer hook supplies u_1=0.5*u_0+0.5*g_1.
        second = [[torch.tensor([3.0, 4.0, 5.0])] for _ in range(3)]
        aggregate = aggregator.aggregate(second)[0]
        momentum = torch.tensor([3.0, 4.0, 5.0])
        expected = first_expected.clone()
        expected[aggregator.last_selected_rows] = momentum[aggregator.last_selected_rows]
        torch.testing.assert_close(aggregator._trackers[0], expected)
        # Identical clients remain identical through ARC and Multi-Krum.
        torch.testing.assert_close(aggregate, expected)

    def test_checkpoint_restores_tracker_and_refresh_rng(self):
        gradients = [[torch.tensor([1.0, 2.0, 3.0])] for _ in range(3)]
        original = self._aggregator(seed=123)
        original.aggregate(gradients)
        state = original.get_state()

        restored = self._aggregator(seed=999)
        restored.set_state(state)
        next_gradients_a = [[torch.tensor([3.0, 2.0, 1.0])] for _ in range(3)]
        next_gradients_b = [[tensor.clone() for tensor in layers] for layers in next_gradients_a]
        output_a = original.aggregate(next_gradients_a)
        output_b = restored.aggregate(next_gradients_b)
        torch.testing.assert_close(output_a[0], output_b[0])
        torch.testing.assert_close(
            original.last_selected_rows, restored.last_selected_rows
        )

    def test_arc_clips_every_tracker_before_aggregation(self):
        aggregator = ProjectedEF21Aggregator(
            m=2,
            r=1,
            k_top=1,
            k_random=1,
            num_byzantine=1,
            byzantine_bound=1,
        )
        trackers = torch.tensor([[3.0, 4.0], [0.0, 2.0], [0.0, 100.0], [1.0, 0.0], [1.0, 1.0]])
        clipped = aggregator._arc(trackers)
        # k_arc=floor(2*(1/5)*4)=1, hence the second-largest norm (5).
        self.assertTrue(torch.all(torch.linalg.vector_norm(clipped, dim=1) <= 5.0))
        torch.testing.assert_close(clipped[0], trackers[0])


class RPEF21AdamTests(unittest.TestCase):
    def test_optimizer_hook_stores_client_momentum_with_exact_initialization(self):
        optimizer = RPEF21Adam(lr=0.1, client_momentum=0.5)
        clients = [
            SimpleNamespace(client_type="Honest"),
            SimpleNamespace(client_type="Byzantine"),
        ]
        context = Context(
            clients=clients,
            grad=[[torch.tensor([2.0])], [torch.tensor([99.0])]],
        )
        optimizer._update_client_momenta(context)
        torch.testing.assert_close(context.grad[0][0], torch.tensor([2.0]))
        self.assertIsNone(optimizer._client_momenta[1])

        context.grad = [[torch.tensor([6.0])], [torch.tensor([88.0])]]
        optimizer._update_client_momenta(context)
        torch.testing.assert_close(context.grad[0][0], torch.tensor([4.0]))
        # Byzantine messages are committed corrections and remain untouched.
        torch.testing.assert_close(context.grad[1][0], torch.tensor([88.0]))

    def test_cap_changes_only_second_moment(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = RPEF21Adam(
            lr=1.0,
            beta1=0.0,
            beta2=0.0,
            eps=1e-8,
            c_max=1.0,
        )
        optimizer.step(SimpleNamespace(model=model), [torch.tensor([[3.0, 4.0]])])

        # z remains [3,4] in m, while v uses its norm-capped [0.6,0.8].
        torch.testing.assert_close(optimizer._m[0], torch.tensor([[3.0, 4.0]]))
        torch.testing.assert_close(optimizer._v_tilde[0], torch.tensor([[0.36, 0.64]]))
        expected = torch.tensor([[-3.0 / 0.6001, -4.0 / 0.8001]])
        torch.testing.assert_close(model.weight, expected)


if __name__ == "__main__":
    unittest.main()
