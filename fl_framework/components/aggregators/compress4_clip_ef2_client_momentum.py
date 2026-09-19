"""AdamK EF2 with honest-client momentum before projected Krum."""

from __future__ import annotations

from typing import List, Optional

import torch

from .compress4_clip_ef2 import OmniscientEF2CompressAggregator


class ClientMomentumOmniscientEF2CompressAggregator(
    OmniscientEF2CompressAggregator
):
    """Apply a per-honest-client EMA before EF2 and projected Krum.

    The processing order is raw honest gradient -> client EMA -> error
    feedback -> internal omniscient attack -> projected Krum -> shared Top-K.
    Consequently the attack is still constructed in the exact message space
    observed by the robust aggregator.
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
        client_momentum: float = 0.9,
        clip_enabled: bool = True,
    ) -> None:
        super().__init__(
            m=m,
            r=r,
            k=k,
            krum_remain=krum_remain,
            byzantine_alpha=byzantine_alpha,
            attack_scale=attack_scale,
            attack_enabled=attack_enabled,
            clip_enabled=clip_enabled,
        )
        if not 0.0 <= client_momentum < 1.0:
            raise ValueError(
                "client_momentum must be in [0, 1), "
                f"got {client_momentum}"
            )
        self.client_momentum = float(client_momentum)
        self._client_momentum_buffers: Optional[List[torch.Tensor]] = None

    @staticmethod
    def _flatten_layers(layers: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([tensor.reshape(-1) for tensor in layers])

    @staticmethod
    def _unflatten_like(
        flat: torch.Tensor, reference: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        result: List[torch.Tensor] = []
        offset = 0
        for tensor in reference:
            count = tensor.numel()
            # EF2 adds the compression residual to its input in place. Return
            # independent tensors so that operation cannot corrupt the EMA.
            result.append(flat[offset : offset + count].view_as(tensor).clone())
            offset += count
        return result

    @torch.no_grad()
    def aggregate(
        self, all_gradients: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        if not all_gradients:
            return []

        num_byzantine = int(self.byzantine_alpha * len(all_gradients))
        num_honest = len(all_gradients) - num_byzantine
        if self._client_momentum_buffers is None:
            self._client_momentum_buffers = [
                torch.zeros_like(self._flatten_layers(all_gradients[index]))
                for index in range(num_honest)
            ]
        elif len(self._client_momentum_buffers) != num_honest:
            raise ValueError(
                "The number of honest clients changed after client-momentum "
                f"initialization: {len(self._client_momentum_buffers)} -> "
                f"{num_honest}."
            )

        processed: List[List[torch.Tensor]] = []
        for index in range(num_honest):
            reference = all_gradients[index]
            flat = self._flatten_layers(reference)
            buffer = self._client_momentum_buffers[index]
            if buffer.device != flat.device or buffer.dtype != flat.dtype:
                buffer = buffer.to(device=flat.device, dtype=flat.dtype)
                self._client_momentum_buffers[index] = buffer
            buffer.mul_(self.client_momentum).add_(
                flat, alpha=1.0 - self.client_momentum
            )
            processed.append(self._unflatten_like(buffer, reference))

        # EF2 replaces Byzantine messages internally when the attack is enabled;
        # preserving these entries also supports no-attack/custom-attack use.
        processed.extend(all_gradients[num_honest:])
        return super().aggregate(processed)

    def get_state(self) -> dict:
        state = super().get_state()
        state["client_momentum"] = self.client_momentum
        state["client_momentum_buffers"] = (
            None
            if self._client_momentum_buffers is None
            else [buffer.detach().cpu() for buffer in self._client_momentum_buffers]
        )
        return state

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        self.client_momentum = state.get(
            "client_momentum", self.client_momentum
        )
        buffers = state.get("client_momentum_buffers")
        self._client_momentum_buffers = (
            None if buffers is None else [buffer.clone() for buffer in buffers]
        )
