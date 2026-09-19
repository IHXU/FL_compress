"""Adam update used by the projected EF21 algorithm in ``new_adamk.md``."""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from fl_framework.core.hooks import Context, HookType, hook_registry

from .base_optimizer import BaseOptimizer


class RPEF21Adam(BaseOptimizer):
    """Use uncapped z for m and globally capped z only for the second moment."""

    def __init__(
        self,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        c_max: float = 10.0,
        client_momentum: float = 0.99,
    ) -> None:
        super().__init__(lr=lr)
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if c_max <= 0.0:
            raise ValueError("c_max must be positive")
        if not 0.0 <= client_momentum < 1.0:
            raise ValueError("client_momentum must be in [0, 1)")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.c_max = float(c_max)
        self.client_momentum = float(client_momentum)
        self._m: Optional[List[torch.Tensor]] = None
        self._v_tilde: Optional[List[torch.Tensor]] = None
        self._client_momenta: Optional[List[Optional[List[torch.Tensor]]]] = None
        self._step = 0

    def register_hooks(self) -> None:
        super().register_hooks()
        hook_registry.register(HookType.BEFORE_AGGREGATE, self._update_client_momenta)

    @torch.no_grad()
    def _update_client_momenta(self, context: Context) -> None:
        """Simulate the local cache from (2.1a) immediately before EF21."""
        if context.clients is None or context.grad is None:
            raise RuntimeError("Client momentum hook requires clients and gradients")
        if self._client_momenta is None:
            self._client_momenta = [None for _ in context.clients]
        elif len(self._client_momenta) != len(context.clients):
            raise ValueError("Number of clients changed after momentum initialization")

        for index, client in enumerate(context.clients):
            if client.client_type == "Byzantine":
                continue
            gradients = context.grad[index]
            if gradients is None:
                raise ValueError("Every honest client must provide a gradient")
            momentum = self._client_momenta[index]
            if momentum is None:
                # The document explicitly initializes u_0 with g_0, not zero.
                momentum = [value.clone() for value in gradients]
            else:
                momentum = self._state_like(momentum, gradients)
                if len(momentum) != len(gradients):
                    raise RuntimeError("Client momentum structure changed")
                torch._foreach_mul_(momentum, self.client_momentum)
                torch._foreach_add_(
                    momentum, gradients, alpha=1.0 - self.client_momentum
                )
            self._client_momenta[index] = momentum
            # The aggregator must see u_t while the raw gradient list remains
            # disposable after this simulated communication step.
            context.grad[index] = [value.clone() for value in momentum]

    @staticmethod
    def _state_like(
        state: List[torch.Tensor], gradients: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        if len(state) != len(gradients):
            raise RuntimeError("Adam state does not match the aggregate structure")
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
        if not aggregated_grad:
            return
        if self._m is None or self._v_tilde is None:
            self._m = [torch.zeros_like(value) for value in aggregated_grad]
            self._v_tilde = [torch.zeros_like(value) for value in aggregated_grad]
        else:
            self._m = self._state_like(self._m, aggregated_grad)
            self._v_tilde = self._state_like(self._v_tilde, aggregated_grad)

        self._step += 1
        torch._foreach_mul_(self._m, self.beta1)
        torch._foreach_add_(self._m, aggregated_grad, alpha=1.0 - self.beta1)

        squared_norm = sum(
            value.square().sum() for value in aggregated_grad
        )
        norm = squared_norm.sqrt()
        clip_factor = min(
            1.0,
            self.c_max / max(norm.item(), torch.finfo(norm.dtype).tiny),
        )
        capped = [value.mul(clip_factor) for value in aggregated_grad]
        torch._foreach_mul_(self._v_tilde, self.beta2)
        torch._foreach_addcmul_(
            self._v_tilde, capped, capped, value=1.0 - self.beta2
        )

        bias_correction2 = 1.0 - self.beta2**self._step
        denominator = [value.div(bias_correction2) for value in self._v_tilde]
        torch._foreach_sqrt_(denominator)
        torch._foreach_add_(denominator, math.sqrt(self.eps))
        torch._foreach_addcdiv_(
            list(server.model.parameters()),
            self._m,
            denominator,
            value=-self.lr,
        )

    def get_state(self) -> dict:
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "c_max": self.c_max,
            "client_momentum": self.client_momentum,
            "step": self._step,
            "m": (
                None
                if self._m is None
                else [value.detach().cpu() for value in self._m]
            ),
            "v_tilde": (
                None
                if self._v_tilde is None
                else [value.detach().cpu() for value in self._v_tilde]
            ),
            "client_momenta": (
                None
                if self._client_momenta is None
                else [
                    None
                    if values is None
                    else [value.detach().cpu() for value in values]
                    for values in self._client_momenta
                ]
            ),
        }

    def set_state(self, state: dict) -> None:
        if not state:
            return
        self.lr = state.get("lr", self.lr)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.eps = state.get("eps", self.eps)
        self.c_max = state.get("c_max", self.c_max)
        self.client_momentum = state.get("client_momentum", self.client_momentum)
        self._step = state.get("step", 0)
        moments = state.get("m")
        variances = state.get("v_tilde")
        self._m = None if moments is None else [value.clone() for value in moments]
        self._v_tilde = (
            None if variances is None else [value.clone() for value in variances]
        )
        client_momenta = state.get("client_momenta")
        self._client_momenta = None if client_momenta is None else [
            None if values is None else [value.clone() for value in values]
            for values in client_momenta
        ]
