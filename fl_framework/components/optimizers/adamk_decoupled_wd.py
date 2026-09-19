"""AdamK with server-side decoupled weight decay.

The original AdamK adds L2 regularization to every honest client's gradient
before projection, robust selection, compression, and error feedback.  This
isolated variant instead applies AdamW-style decay directly to the global
model, so the decay cannot change Krum scores, the Top-K mask, or EF state.
"""

from __future__ import annotations

import torch

from .adamk import ADAMK


class DecoupledWDADAMK(ADAMK):
    """AdamK whose only weight decay is ``param *= 1 - lr * wd``."""

    def __init__(
        self,
        lr: float,
        decoupled_weight_decay: float = 1.0e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1.0e-8,
        lr_decay_rate: float = 1.0,
        lr_decay_milestone: list[int] | None = None,
    ) -> None:
        if decoupled_weight_decay < 0:
            raise ValueError("decoupled_weight_decay must be non-negative")
        super().__init__(
            lr=lr,
            weight_decay=0.0,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            lr_decay_rate=lr_decay_rate,
            lr_decay_milestone=(
                [5, 7] if lr_decay_milestone is None else lr_decay_milestone
            ),
        )
        self.decoupled_weight_decay = decoupled_weight_decay

    @torch.no_grad()
    def step(self, server, aggregated_grad) -> None:
        if server is None or aggregated_grad is None:
            return super().step(server, aggregated_grad)
        decay_factor = 1.0 - self.lr * self.decoupled_weight_decay
        torch._foreach_mul_(list(server.model.parameters()), decay_factor)
        super().step(server, aggregated_grad)

    def get_state(self) -> dict:
        state = super().get_state()
        state["decoupled_weight_decay"] = self.decoupled_weight_decay
        return state

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        self.decoupled_weight_decay = state.get(
            "decoupled_weight_decay", self.decoupled_weight_decay
        )
