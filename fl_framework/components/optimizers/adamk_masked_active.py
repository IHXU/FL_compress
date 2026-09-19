"""Mask-aware AdamK that updates parameters only on active Top-K rows."""

from __future__ import annotations

from typing import List, Optional

import torch

from .adamk import ADAMK
from fl_framework.core.hooks import Context, HookType, hook_registry


class MaskedActiveADAMK(ADAMK):
    """Decay all moments normally, but update only active Top-K parameters.

    For inactive coordinates the zero-gradient Adam recurrence is applied:
    ``m <- beta1*m`` and ``v <- beta2*v``. Their parameters are not changed.
    The exact row mask is captured from the compatible CompressAggregator after
    aggregation, so naturally zero entries inside a selected row remain active.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._active_masks: Optional[List[torch.Tensor]] = None

    def register_hooks(self) -> None:
        super().register_hooks()
        hook_registry.register(HookType.AFTER_AGGREGATE, self._capture_active_mask)

    @torch.no_grad()
    def _capture_active_mask(self, context: Context) -> None:
        aggregator = context.aggregator
        selected_rows = getattr(aggregator, "last_selected_rows", None)
        row_count = getattr(aggregator, "m", None)
        row_width = getattr(aggregator, "n", None)
        gradients = context.aggregated_grad
        if selected_rows is None or row_count is None or row_width is None:
            raise RuntimeError(
                "MaskedActiveADAMK requires an aggregator exposing the exact "
                "last_selected_rows, m, and n."
            )
        if not gradients:
            self._active_masks = None
            return

        device = gradients[0].device
        row_mask = torch.zeros(row_count, dtype=torch.bool, device=device)
        row_mask[selected_rows.to(device=device)] = True
        flat_mask = row_mask.repeat_interleave(row_width)
        expected = sum(gradient.numel() for gradient in gradients)
        if flat_mask.numel() != expected:
            raise RuntimeError(
                f"Top-K mask has {flat_mask.numel()} entries, expected {expected}."
            )

        masks: List[torch.Tensor] = []
        offset = 0
        for gradient in gradients:
            count = gradient.numel()
            masks.append(flat_mask[offset : offset + count].view_as(gradient))
            offset += count
        self._active_masks = masks

    @staticmethod
    def _state_on_gradient(
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
        if self._active_masks is None:
            raise RuntimeError(
                "MaskedActiveADAMK did not receive a Top-K mask before step()."
            )
        if len(self._active_masks) != len(aggregated_grad):
            raise RuntimeError("Top-K mask does not match gradient structure.")

        if self._m is None or self._v is None:
            # Preserve legacy AdamK's first-active-step initialization.
            self._m = [gradient.detach().clone() for gradient in aggregated_grad]
            self._v = [gradient.detach().square() for gradient in aggregated_grad]
        else:
            if len(self._m) != len(aggregated_grad):
                raise RuntimeError(
                    "MaskedActiveADAMK checkpoint state does not match the "
                    "aggregated gradient structure."
                )
            self._m = self._state_on_gradient(self._m, aggregated_grad)
            self._v = self._state_on_gradient(self._v, aggregated_grad)
            torch._foreach_mul_(self._m, self.beta1)
            torch._foreach_add_(
                self._m, aggregated_grad, alpha=1.0 - self.beta1
            )
            torch._foreach_mul_(self._v, self.beta2)
            torch._foreach_addcmul_(
                self._v,
                aggregated_grad,
                aggregated_grad,
                value=1.0 - self.beta2,
            )

        parameters = list(server.model.parameters())
        for parameter, moment, variance, mask in zip(
            parameters, self._m, self._v, self._active_masks
        ):
            if not torch.any(mask):
                continue
            denominator = variance[mask].sqrt().add_(self.eps)
            # Boolean advanced indexing returns a copy, so use indexed
            # assignment to ensure the original parameter is modified.
            parameter[mask] = parameter[mask] - self.lr * (
                moment[mask] / denominator
            )
