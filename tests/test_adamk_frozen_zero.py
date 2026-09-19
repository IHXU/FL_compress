import unittest
from types import SimpleNamespace

import torch

from fl_framework.components.optimizers.adamk_frozen_zero import FrozenZeroADAMK


class FrozenZeroADAMKTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.model.weight.zero_()
        self.server = SimpleNamespace(model=self.model)
        self.optimizer = FrozenZeroADAMK(
            lr=0.01, beta1=0.9, beta2=0.99, eps=1e-8
        )

    def test_zero_coordinates_keep_both_moments(self):
        self.optimizer.step(self.server, [torch.tensor([[2.0, 3.0, 0.0]])])
        old_m = self.optimizer._m[0].clone()
        old_v = self.optimizer._v[0].clone()

        self.optimizer.step(self.server, [torch.tensor([[0.0, 4.0, 0.0]])])

        torch.testing.assert_close(self.optimizer._m[0][0, 0], old_m[0, 0])
        torch.testing.assert_close(self.optimizer._v[0][0, 0], old_v[0, 0])
        torch.testing.assert_close(self.optimizer._m[0][0, 2], old_m[0, 2])
        torch.testing.assert_close(self.optimizer._v[0][0, 2], old_v[0, 2])
        torch.testing.assert_close(
            self.optimizer._m[0][0, 1], torch.tensor(3.1)
        )
        torch.testing.assert_close(
            self.optimizer._v[0][0, 1], torch.tensor(9.07)
        )

    def test_all_zero_gradient_leaves_state_unchanged(self):
        self.optimizer.step(self.server, [torch.tensor([[1.0, -2.0, 3.0]])])
        old_m = self.optimizer._m[0].clone()
        old_v = self.optimizer._v[0].clone()

        self.optimizer.step(self.server, [torch.zeros(1, 3)])

        torch.testing.assert_close(self.optimizer._m[0], old_m)
        torch.testing.assert_close(self.optimizer._v[0], old_v)


if __name__ == "__main__":
    unittest.main()
