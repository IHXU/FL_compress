"""Isolated optimizers for the AdamK error-feedback ablation."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from fl_framework.core.hooks import Context, HookType, hook_registry

from .adamk import ADAMK
from .base_optimizer import BaseOptimizer


class LegacyADAMK(ADAMK):
    """Named, isolated entry point for experiments A and B."""


class BiasCorrectedADAMK(BaseOptimizer):
    """Standard bias-corrected Adam update with AdamK's existing hooks.

    Weight decay intentionally remains coupled and client-side in this
    experiment so C differs from B only in the Adam state/update equations.
    The step counter advances once per optimizer update (300 times per round in
    these configurations), not once per communication round.
    """

    def __init__(
        self,
        lr: float,
        weight_decay: float = 0.00005,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        lr_decay_rate: float = 1.0,
        lr_decay_milestone: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(lr=lr)
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_milestone = [0] + list(lr_decay_milestone or [5, 7])
        self.base_lr = lr
        self._m: Optional[List[torch.Tensor]] = None
        self._v: Optional[List[torch.Tensor]] = None
        self._step = 0

    def register_hooks(self) -> None:
        super().register_hooks()
        hook_registry.register(HookType.AFTER_COMPUTE, self._weight_decay)
        hook_registry.register(HookType.BEFORE_UPDATE, self.get_lr)

    def get_lr(self, context: Context) -> None:
        current_round = context.current_round + 1
        decay_index = 0
        for i in range(len(self.lr_decay_milestone) - 1, -1, -1):
            if current_round >= self.lr_decay_milestone[i]:
                decay_index = i
                break
        self.lr = self.base_lr * self.lr_decay_rate**decay_index

    @torch.no_grad()
    def _weight_decay(self, context: Context) -> None:
        client_id = context.current_client_id
        client = context.clients[client_id]
        if client.client_type != "Byzantine":
            for grad, param in zip(context.grad[client_id], client.model.parameters()):
                if grad is not None:
                    grad.add_(param, alpha=self.weight_decay)

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

        if self._m is None or self._v is None:
            self._m = [torch.zeros_like(g) for g in aggregated_grad]
            self._v = [torch.zeros_like(g) for g in aggregated_grad]
        else:
            self._m = self._state_on_gradient(self._m, aggregated_grad)
            self._v = self._state_on_gradient(self._v, aggregated_grad)

        self._step += 1
        torch._foreach_mul_(self._m, self.beta1)
        torch._foreach_add_(self._m, aggregated_grad, alpha=1.0 - self.beta1)
        torch._foreach_mul_(self._v, self.beta2)
        torch._foreach_addcmul_(
            self._v,
            aggregated_grad,
            aggregated_grad,
            value=1.0 - self.beta2,
        )

        bias_correction1 = 1.0 - self.beta1**self._step
        bias_correction2 = 1.0 - self.beta2**self._step
        bias_correction2_sqrt = bias_correction2**0.5
        step_size = self.lr * bias_correction2_sqrt / bias_correction1

        denominator = [value.clone() for value in self._v]
        torch._foreach_sqrt_(denominator)
        # With the folded step size above, epsilon must be scaled by
        # sqrt(1-beta2**t) to remain exactly equivalent to m_hat/v_hat Adam.
        torch._foreach_add_(denominator, self.eps * bias_correction2_sqrt)
        torch._foreach_addcdiv_(
            list(server.model.parameters()),
            self._m,
            denominator,
            value=-step_size,
        )

    def get_state(self) -> dict:
        return {
            "lr": self.lr,
            "base_lr": self.base_lr,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "lr_decay_rate": self.lr_decay_rate,
            "lr_decay_milestone": self.lr_decay_milestone,
            "step": self._step,
            "m": None if self._m is None else [value.detach().cpu() for value in self._m],
            "v": None if self._v is None else [value.detach().cpu() for value in self._v],
        }

    def set_state(self, state: dict) -> None:
        self.lr = state.get("lr", self.lr)
        self.base_lr = state.get("base_lr", self.base_lr)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.eps = state.get("eps", self.eps)
        self.lr_decay_rate = state.get("lr_decay_rate", self.lr_decay_rate)
        self.lr_decay_milestone = state.get(
            "lr_decay_milestone", self.lr_decay_milestone
        )
        self._step = state.get("step", 0)
        moments = state.get("m")
        variances = state.get("v")
        self._m = None if moments is None else [value.clone() for value in moments]
        self._v = None if variances is None else [value.clone() for value in variances]
