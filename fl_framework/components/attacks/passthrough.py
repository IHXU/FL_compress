"""Benign placeholder used when an aggregator constructs the attack message.

The Byzantine client computes its ordinary local gradient so every optimizer
hook can run normally.  The Krum-space omniscient aggregators replace the
Byzantine entries only after all method-specific momentum/compression hooks
have produced the exact messages that Krum would otherwise receive.
"""

from typing import List

import torch

from .base_attack import BaseAttack, Context


class PassthroughAttack(BaseAttack):
    """Return the client's current gradient without modifying it."""

    if_byz_compute_grad = True

    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        client_id = context.current_client_id
        return [gradient.clone() for gradient in context.grad[client_id]]
