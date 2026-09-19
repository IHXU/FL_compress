"""Krum-space omniscient attack without changing legacy Krum behaviour.

This component is intentionally opt-in.  Existing configurations continue to
instantiate ``KrumAggregator`` and are therefore unaffected.
"""

from __future__ import annotations

from typing import List

import torch

from .krum import KrumAggregator


class OmniscientKrumAggregator(KrumAggregator):
    """Replace Byzantine Krum inputs with a flipped honest-message mean.

    The replacement happens inside ``aggregate``.  Consequently every
    optimizer hook has already converted raw client gradients into the exact
    representation used by Krum: momentum, compressed vector, reconstructed
    estimator, or any other method-specific message.

    Clients are ordered as honest clients followed by Byzantine clients by the
    framework.  ``attack_scale=1`` sends the negative honest mean; larger
    values strengthen the same full-knowledge direction.
    """

    def __init__(
        self,
        num_byzantine: int,
        attack_scale: float = 1.0,
        attack_enabled: bool = True,
    ) -> None:
        super().__init__(num_byzantine=num_byzantine)
        if attack_scale < 0:
            raise ValueError("attack_scale must be non-negative")
        self.attack_scale = float(attack_scale)
        self.attack_enabled = bool(attack_enabled)

    @staticmethod
    def _mean_messages(
        honest_messages: List[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        if not honest_messages:
            raise ValueError("The omniscient attack requires honest messages")
        layer_count = len(honest_messages[0])
        if any(len(message) != layer_count for message in honest_messages):
            raise ValueError("All Krum messages must have the same tensor structure")

        means: List[torch.Tensor] = []
        for layer_index in range(layer_count):
            reference = honest_messages[0][layer_index]
            total = torch.zeros_like(reference)
            for message in honest_messages:
                tensor = message[layer_index]
                total.add_(
                    tensor.to(device=reference.device, dtype=reference.dtype)
                    if tensor.device != reference.device or tensor.dtype != reference.dtype
                    else tensor
                )
            means.append(total.div_(len(honest_messages)))
        return means

    @torch.no_grad()
    def aggregate(
        self, all_gradients: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        if not all_gradients:
            return []
        if not self.attack_enabled or self.byzantine_size == 0:
            return super().aggregate(all_gradients)
        if self.byzantine_size >= len(all_gradients):
            raise ValueError("At least one honest Krum message is required")

        num_honest = len(all_gradients) - self.byzantine_size
        honest_messages = all_gradients[:num_honest]
        honest_mean = self._mean_messages(honest_messages)
        malicious_message = [
            tensor.mul(-self.attack_scale) for tensor in honest_mean
        ]

        attacked_messages = list(all_gradients[:num_honest])
        for _ in range(self.byzantine_size):
            attacked_messages.append([tensor.clone() for tensor in malicious_message])
        return super().aggregate(attacked_messages)

    def get_state(self) -> dict:
        return {
            "num_byzantine": self.byzantine_size,
            "attack_scale": self.attack_scale,
            "attack_enabled": self.attack_enabled,
        }

    def set_state(self, state: dict) -> None:
        if not state:
            return
        self.byzantine_size = state.get("num_byzantine", self.byzantine_size)
        self.attack_scale = state.get("attack_scale", self.attack_scale)
        self.attack_enabled = state.get("attack_enabled", self.attack_enabled)
