"""AdamK variant that preserves moment state at compressed zero coordinates.

This module is intentionally separate from :mod:`adamk` so existing
experiments and checkpoints retain their original optimizer behaviour.
"""

from __future__ import annotations

from typing import List

import torch

from .adamk import ADAMK


class FrozenZeroADAMK(ADAMK):
    """Legacy AdamK with sparse, mask-aware moment updates.

    Top-k compression produces exact zeros outside the selected coordinates.
    For a compressed gradient ``g`` and active mask ``a = (g != 0)``, this
    optimizer applies the usual AdamK recurrence only where ``a`` is true::

        m[a] = beta1 * m[a] + (1 - beta1) * g[a]
        v[a] = beta2 * v[a] + (1 - beta2) * g[a] ** 2

    Both moments remain exactly unchanged where ``a`` is false.  The model
    update still uses the full stored moments, including values retained from
    earlier steps at currently inactive coordinates.
    """

    @staticmethod
    def _move_state_to_gradients(
        state: List[torch.Tensor], gradients: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        return [
            value.to(device=gradient.device, dtype=gradient.dtype)
            if value.device != gradient.device or value.dtype != gradient.dtype
            else value
            for value, gradient in zip(state, gradients)
        ]

    @torch.no_grad()
    def step(self, server, aggregated_grad: List[torch.Tensor]) -> None:
        if server is None or aggregated_grad is None:
            return

        if self._m is None or self._v is None:
            # Match legacy ADAMK initialization exactly.  Compressed zero
            # coordinates consequently start with zero moment state.
            self._m = [gradient.detach().clone() for gradient in aggregated_grad]
            self._v = [gradient.detach().square() for gradient in aggregated_grad]
        else:
            if len(self._m) != len(aggregated_grad) or len(self._v) != len(
                aggregated_grad
            ):
                raise RuntimeError(
                    "FrozenZeroADAMK checkpoint state does not match the "
                    "aggregated gradient structure."
                )

            self._m = self._move_state_to_gradients(self._m, aggregated_grad)
            self._v = self._move_state_to_gradients(self._v, aggregated_grad)

            for moment, variance, gradient in zip(
                self._m, self._v, aggregated_grad
            ):
                active = gradient.ne(0)
                moment[active] = (
                    moment[active] * self.beta1
                    + gradient[active] * (1.0 - self.beta1)
                )
                variance[active] = (
                    variance[active] * self.beta2
                    + gradient[active].square() * (1.0 - self.beta2)
                )

        parameters = list(server.model.parameters())
        denominator = [variance.clone() for variance in self._v]
        torch._foreach_sqrt_(denominator)
        torch._foreach_add_(denominator, self.eps)
        torch._foreach_addcdiv_(
            parameters, self._m, denominator, value=-self.lr
        )
