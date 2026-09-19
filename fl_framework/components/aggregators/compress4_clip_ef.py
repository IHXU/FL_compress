"""Isolated aggregators for the AdamK error-feedback ablation.

This module deliberately leaves ``compress4_clip.py`` unchanged so experiments
that are already running keep using exactly the code they imported at startup.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .compress4_clip import CompressAggregator


class LegacyCompressAggregator(CompressAggregator):
    """Named, isolated entry point for experiment A (no error feedback)."""


class ErrorFeedbackCompressAggregator(CompressAggregator):
    """ARC-Top-K aggregation with client-side compression error feedback.

    The Krum retention ratio and Top-K compression ratio are inherited without
    modification.  A residual is maintained only for the clients presumed to
    be honest (the framework always places Byzantine clients last).  Byzantine
    gradients are passed to the original robust aggregator unchanged.

    For honest client ``i`` the compressor input and residual update are::

        u_i = g_i + e_i
        c_i = mask_I(u_i)
        e_i = u_i - c_i

    ``I`` is the same global row mask selected by the original aggregator.
    With the ablation's ``krum_remain=0.05`` exactly one client is retained, so
    nonzero rows in the returned aggregate identify ``I`` exactly (a selected
    all-zero row has zero compression error and is harmless).
    """

    def __init__(
        self,
        m: int,
        r: int,
        k: int,
        krum_remain: float,
        byzantine_alpha: float,
        clip_enabled: bool = True,
    ) -> None:
        super().__init__(
            m, r, k, krum_remain, byzantine_alpha, clip_enabled=clip_enabled
        )
        self._error_residuals: Optional[List[torch.Tensor]] = None
        self._num_honest_residuals: Optional[int] = None

    def _ensure_residuals(
        self, all_gradients: List[List[torch.Tensor]]
    ) -> int:
        num_clients = len(all_gradients)
        num_byzantine = int(self.byzantine_alpha * num_clients)
        num_honest = num_clients - num_byzantine

        if self._error_residuals is None:
            self._error_residuals = [
                torch.zeros(
                    sum(g.numel() for g in all_gradients[i]),
                    device=all_gradients[i][0].device,
                    dtype=all_gradients[i][0].dtype,
                )
                for i in range(num_honest)
            ]
            self._num_honest_residuals = num_honest
        elif self._num_honest_residuals != num_honest:
            raise ValueError(
                "The number of honest clients changed after error-feedback "
                f"initialization: {self._num_honest_residuals} -> {num_honest}."
            )

        return num_honest

    @staticmethod
    def _add_flat_residual_(
        gradient_layers: List[torch.Tensor], residual: torch.Tensor
    ) -> None:
        offset = 0
        for gradient in gradient_layers:
            count = gradient.numel()
            gradient.add_(residual[offset : offset + count].view_as(gradient))
            offset += count

    @torch.no_grad()
    def aggregate(
        self, all_gradients: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        if not all_gradients:
            return []

        num_honest = self._ensure_residuals(all_gradients)
        assert self._error_residuals is not None

        # Gradients in Context are discarded after this step, so adding the
        # residual in place avoids another full model-sized copy per client.
        for client_idx in range(num_honest):
            residual = self._error_residuals[client_idx]
            reference = all_gradients[client_idx][0]
            if residual.device != reference.device or residual.dtype != reference.dtype:
                residual = residual.to(device=reference.device, dtype=reference.dtype)
                self._error_residuals[client_idx] = residual
            self._add_flat_residual_(all_gradients[client_idx], residual)

        aggregated = super().aggregate(all_gradients)
        if not aggregated:
            return aggregated

        flat_aggregated = self._flatten_gradients(aggregated)
        selected_rows = flat_aggregated.reshape(self.m, self.n).ne(0).any(dim=1)

        # Keep precisely the part omitted by the shared row compressor.  A
        # robustly rejected client does not accumulate its transmitted rows;
        # error feedback compensates compression, not Krum rejection.
        for client_idx in range(num_honest):
            corrected = self._flatten_gradients(all_gradients[client_idx])
            residual = self._error_residuals[client_idx]
            residual.copy_(corrected)
            residual.reshape(self.m, self.n)[selected_rows] = 0

        return aggregated

    def get_state(self) -> dict:
        state = super().get_state()
        state["error_residuals"] = (
            None
            if self._error_residuals is None
            else [residual.detach().cpu() for residual in self._error_residuals]
        )
        state["num_honest_residuals"] = self._num_honest_residuals
        return state

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        residuals = state.get("error_residuals")
        self._error_residuals = (
            None if residuals is None else [residual.clone() for residual in residuals]
        )
        self._num_honest_residuals = state.get("num_honest_residuals")
