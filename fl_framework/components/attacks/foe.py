from typing import List
import torch
from .base_attack import BaseAttack, Context

class FOE(BaseAttack):
    """
    Implements the FOE attack.

    This attack manipulates a malicious client's gradient update by scaling
    the mean gradient (calculated over all honest clients) by a negative factor `mu`.
    This effectively flips the sign of the true gradient and may scale its magnitude,
    aiming to steer the global model update in the opposite and potentially disruptive direction.
    """

    def __init__(self, mu: float = -10):
        """
        Initializes the FOE attack.

        for Krum mu use e.g. -0.1 to 1 (abs small), for CwMed use e.g.-10 (abs large)

        Args:
            mu (float): The scaling factor to apply to the mean gradient.
                        Typically negative to flip the sign; magnitude controls attack strength.
        """

        self.mu = mu
        self.mean_grad = []
        self.current_step = None

    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        """
        Generates the malicious gradient by flipping and scaling the sign of the mean honest gradient.

        For each training step, this method computes and caches the mean gradient from all honest clients.
        It then returns this mean gradient scaled by `mu` as the malicious update.

        Args:
            context (Context): The attack context containing the current step and the list of honest clients' gradients.

        Returns:
            List[torch.Tensor]: The list of malicious gradients for each layer, after sign flipping and scaling.
        """
        # Under partial-participation optimizers (e.g. DByzSGDM with
        # participation < 1), honest clients that are not selected this step
        # have grad=None. Filter them out so the mean is computed over the
        # honest clients that actually produced a raw gradient.
        honest_grads = [g for g in context.all_honest_gradients if g is not None]

        if not honest_grads:
            # No honest client participated this step. Reuse the last cached
            # mean gradient as a delayed reference (consistent with the
            # delayed-momentum design of DByzSGDM); on the very first step
            # fall back to a zero vector of the correct shape.
            if not self.mean_grad:
                return [
                    torch.zeros_like(p) * self.mu
                    for p in context.server.model.parameters()
                ]
            return [g * self.mu for g in self.mean_grad]

        ref_grad = honest_grads[0]
        if self.current_step != context.current_step:
            self.current_step = context.current_step
            self.mean_grad = []

            num_layers = len(ref_grad)

            # Iterate through each layer/tensor position
            for i in range(num_layers):
                # Collect the i-th gradient tensor from all clients
                layer_gradients = [client_grad[i] for client_grad in honest_grads]

                # Store original shape and device for reconstruction
                original_shape = layer_gradients[0].shape
                device = layer_gradients[0].device

                sum_tensor = torch.zeros_like(layer_gradients[0], device=device)
                for g in layer_gradients:
                    if g.device != device:
                        sum_tensor.add_(g.to(device))
                    else:
                        sum_tensor.add_(g)

                mean_tensor = sum_tensor.div_(len(layer_gradients))
                self.mean_grad.append(mean_tensor.view(original_shape))

        return [g * self.mu for g in self.mean_grad]
