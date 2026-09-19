import unittest
from types import SimpleNamespace

import torch

from fl_framework.components.aggregators.compress4_clip_ef2_client_momentum import (
    ClientMomentumOmniscientEF2CompressAggregator,
)
from fl_framework.components.optimizers.adamk_masked_active import (
    MaskedActiveADAMK,
)
from fl_framework.core.hooks import Context


class MaskedActiveADAMKTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.model.weight.zero_()
        self.server = SimpleNamespace(model=self.model)
        self.optimizer = MaskedActiveADAMK(
            lr=0.01, beta1=0.9, beta2=0.99, eps=1e-8
        )

    def test_inactive_moments_decay_but_parameters_do_not_update(self):
        self.optimizer._active_masks = [
            torch.tensor([[True, False, True]])
        ]
        self.optimizer.step(
            self.server, [torch.tensor([[1.0, 2.0, 3.0]])]
        )
        first_parameters = self.model.weight.detach().clone()
        torch.testing.assert_close(
            first_parameters, torch.tensor([[-0.01, 0.0, -0.01]])
        )

        self.optimizer._active_masks = [
            torch.tensor([[False, True, False]])
        ]
        self.optimizer.step(
            self.server, [torch.tensor([[0.0, 4.0, 0.0]])]
        )

        # Coordinates 0 and 2 decay their state but remain unchanged in-model.
        torch.testing.assert_close(self.optimizer._m[0][0, 0], torch.tensor(0.9))
        torch.testing.assert_close(self.optimizer._v[0][0, 0], torch.tensor(0.99))
        torch.testing.assert_close(self.model.weight[0, 0], first_parameters[0, 0])
        torch.testing.assert_close(self.model.weight[0, 2], first_parameters[0, 2])
        self.assertLess(self.model.weight[0, 1].item(), 0.0)

    def test_captures_exact_row_mask_from_aggregator(self):
        aggregator = SimpleNamespace(
            last_selected_rows=torch.tensor([0, 2]), m=3, n=1
        )
        context = Context(
            aggregator=aggregator,
            aggregated_grad=[torch.tensor([[0.0, 5.0, 0.0]])],
        )
        self.optimizer._capture_active_mask(context)
        torch.testing.assert_close(
            self.optimizer._active_masks[0],
            torch.tensor([[True, False, True]]),
        )


class ClientMomentumEF2Tests(unittest.TestCase):
    def test_ef_does_not_mutate_client_momentum_buffer(self):
        aggregator = ClientMomentumOmniscientEF2CompressAggregator(
            m=2,
            r=1,
            k=1,
            krum_remain=0.5,
            byzantine_alpha=0.0,
            attack_enabled=False,
            client_momentum=0.9,
        )
        gradients = [
            [torch.tensor([1.0, 2.0])],
            [torch.tensor([3.0, 4.0])],
        ]
        aggregator.aggregate(gradients)
        torch.testing.assert_close(
            aggregator._client_momentum_buffers[0], torch.tensor([0.1, 0.2])
        )
        torch.testing.assert_close(
            aggregator._client_momentum_buffers[1], torch.tensor([0.3, 0.4])
        )


if __name__ == "__main__":
    unittest.main()
