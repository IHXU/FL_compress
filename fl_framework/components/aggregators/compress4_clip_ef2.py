"""AdamK EF2 with an attack defined in its internal Multi-Krum space.

This file is separate from ``compress4_clip.py`` and ``compress4_clip_ef.py``
so all historical configurations and checkpoints keep their old behaviour.
"""

from __future__ import annotations

from typing import List

import torch

from .compress4_clip import CompressAggregator
from .compress4_clip_ef import ErrorFeedbackCompressAggregator


class OmniscientEF2CompressAggregator(ErrorFeedbackCompressAggregator):
    """Error feedback plus honest-projection omniscient Sign Flipping.

    Honest residuals are added before the attack is constructed.  The forged
    full-gradient message is the negative mean of those corrected honest
    messages.  Because AdamK's projection is linear and shared by all clients,
    its internal Multi-Krum representation is exactly the negative mean of the
    honest projected representations.  Thus the attack is defined using what
    Multi-Krum actually receives, not raw client gradients.
    """

    def __init__(
        self,
        m: int,
        r: int,
        k: int,
        krum_remain: float,
        byzantine_alpha: float,
        attack_scale: float = 1.0,
        attack_enabled: bool = True,
        clip_enabled: bool = True,
    ) -> None:
        super().__init__(
            m,
            r,
            k,
            krum_remain,
            byzantine_alpha,
            clip_enabled=clip_enabled,
        )
        if attack_scale < 0:
            raise ValueError("attack_scale must be non-negative")
        self.attack_scale = float(attack_scale)
        self.attack_enabled = bool(attack_enabled)

    @staticmethod
    def _mean_gradient_lists(
        honest_gradients: List[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        if not honest_gradients:
            raise ValueError("EF2 requires at least one honest client")
        layer_count = len(honest_gradients[0])
        means: List[torch.Tensor] = []
        for layer_index in range(layer_count):
            reference = honest_gradients[0][layer_index]
            total = torch.zeros_like(reference)
            for gradients in honest_gradients:
                tensor = gradients[layer_index]
                total.add_(
                    tensor.to(device=reference.device, dtype=reference.dtype)
                    if tensor.device != reference.device or tensor.dtype != reference.dtype
                    else tensor
                )
            means.append(total.div_(len(honest_gradients)))
        return means

    @torch.no_grad()
    def aggregate(
        self, all_gradients: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        if not all_gradients:
            return []

        num_honest = self._ensure_residuals(all_gradients)
        assert self._error_residuals is not None

        # EF2 correction is part of the honest message observed by the
        # omniscient attacker and by AdamK's projected Multi-Krum.
        for client_index in range(num_honest):
            residual = self._error_residuals[client_index]
            reference = all_gradients[client_index][0]
            if residual.device != reference.device or residual.dtype != reference.dtype:
                residual = residual.to(device=reference.device, dtype=reference.dtype)
                self._error_residuals[client_index] = residual
            self._add_flat_residual_(all_gradients[client_index], residual)

        attacked_gradients = list(all_gradients[:num_honest])
        num_byzantine = len(all_gradients) - num_honest
        if self.attack_enabled and num_byzantine > 0:
            honest_mean = self._mean_gradient_lists(attacked_gradients)
            malicious = [
                tensor.mul(-self.attack_scale) for tensor in honest_mean
            ]
            for _ in range(num_byzantine):
                attacked_gradients.append([tensor.clone() for tensor in malicious])
        else:
            attacked_gradients.extend(all_gradients[num_honest:])

        # Call the legacy compressor directly: the EF correction has already
        # been applied above, so calling the EF1 override would add it twice.
        aggregated = CompressAggregator.aggregate(self, attacked_gradients)
        if not aggregated:
            return aggregated

        flat_aggregated = self._flatten_gradients(aggregated)
        selected_rows = flat_aggregated.reshape(self.m, self.n).ne(0).any(dim=1)

        # Keep the rows omitted by the shared row compressor as the next EF2
        # residual.  Only honest clients own residual state.
        for client_index in range(num_honest):
            corrected = self._flatten_gradients(attacked_gradients[client_index])
            residual = self._error_residuals[client_index]
            residual.copy_(corrected)
            residual.reshape(self.m, self.n)[selected_rows] = 0

        return aggregated

    def get_state(self) -> dict:
        state = super().get_state()
        state["attack_scale"] = self.attack_scale
        state["attack_enabled"] = self.attack_enabled
        state["ef_version"] = 2
        return state

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        self.attack_scale = state.get("attack_scale", self.attack_scale)
        self.attack_enabled = state.get("attack_enabled", self.attack_enabled)
