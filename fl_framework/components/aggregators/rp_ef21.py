"""Projected EF21 tracker aggregation for the modified AdamK algorithm.

This component is intentionally independent from ``compress4_clip_ef*``.
Those aggregators implement classical error feedback, whereas this module
keeps a full EF21 tracker ``q`` for every client.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch

from .base_aggregator import BaseAggregator


class ProjectedEF21Aggregator(BaseAggregator):
    """Simulate Algorithm 1's client/server communication and tracker state.

    Honest clients are assumed to precede Byzantine clients, which is the
    ordering used by ``run_experiment.py``. The optimizer hook has already
    replaced each honest stochastic gradient by its client momentum ``u_t``.
    Incoming Byzantine tensors are interpreted as committed corrections.
    """

    def __init__(
        self,
        m: int,
        r: int,
        k_top: int,
        k_random: int,
        num_byzantine: int,
        byzantine_bound: Optional[int] = None,
        correction_selection_size: Optional[int] = None,
        tracker_selection_size: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        if m < 2:
            raise ValueError("m must be at least 2")
        if r < 1:
            raise ValueError("r must be positive")
        if not 1 <= k_top < m:
            raise ValueError("k_top must satisfy 1 <= k_top < m")
        if not 1 <= k_random <= m - k_top:
            raise ValueError("k_random must satisfy 1 <= k_random <= m-k_top")
        if num_byzantine < 0:
            raise ValueError("num_byzantine cannot be negative")
        self.m = int(m)
        self.r = int(r)
        self.k_top = int(k_top)
        self.k_random = int(k_random)
        self.num_byzantine = int(num_byzantine)
        self.byzantine_bound = int(
            num_byzantine if byzantine_bound is None else byzantine_bound
        )
        if self.byzantine_bound < self.num_byzantine:
            raise ValueError(
                "byzantine_bound cannot be smaller than num_byzantine"
            )
        self.correction_selection_size = correction_selection_size
        self.tracker_selection_size = tracker_selection_size
        self.seed = int(seed)

        self.d: Optional[int] = None
        self.n: Optional[int] = None
        self.reference_shapes: Optional[List[torch.Size]] = None
        self._trackers: Optional[List[torch.Tensor]] = None
        self.last_selected_rows: Optional[torch.Tensor] = None
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)

    @staticmethod
    def _flatten(layers: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat([tensor.reshape(-1) for tensor in layers])

    @staticmethod
    def _unflatten(
        flat: torch.Tensor, shapes: Sequence[torch.Size]
    ) -> List[torch.Tensor]:
        result: List[torch.Tensor] = []
        offset = 0
        for shape in shapes:
            count = shape.numel()
            result.append(flat[offset : offset + count].view(shape))
            offset += count
        if offset != flat.numel():
            raise ValueError("Stored layer shapes do not match the tracker dimension")
        return result

    @staticmethod
    def _move_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if tensor.device != reference.device or tensor.dtype != reference.dtype:
            return tensor.to(device=reference.device, dtype=reference.dtype)
        return tensor

    def _validate_and_initialize(
        self, all_gradients: List[List[torch.Tensor]]
    ) -> tuple[List[torch.Tensor], int, int]:
        total_clients = len(all_gradients)
        if total_clients <= 2 * self.byzantine_bound + 2:
            raise ValueError(
                "Projected Multi-Krum requires N >= 2*byzantine_bound + 3"
            )
        if self.num_byzantine >= total_clients:
            raise ValueError("At least one honest client is required")
        if any(not layers for layers in all_gradients):
            raise ValueError("Every client message must contain at least one tensor")

        num_honest = total_clients - self.num_byzantine
        reference = all_gradients[0][0]
        flat_messages = [
            self._move_like(self._flatten(layers), reference)
            for layers in all_gradients
        ]
        dimension = flat_messages[0].numel()
        if any(message.numel() != dimension for message in flat_messages):
            raise ValueError("All client messages must have the same dimension")

        shapes = [tensor.shape for tensor in all_gradients[0]]
        if self.d is None:
            if dimension % self.m != 0:
                raise ValueError(
                    f"Gradient dimension {dimension} is not divisible by m={self.m}"
                )
            self.d = dimension
            self.n = dimension // self.m
            self.reference_shapes = shapes
            # EF21 specifies q_{-1}=0 for every client.
            self._trackers = [
                torch.zeros(dimension, device=reference.device, dtype=reference.dtype)
                for _ in range(total_clients)
            ]
        elif dimension != self.d or shapes != self.reference_shapes:
            raise ValueError("Client message structure changed after EF21 initialization")
        elif self._trackers is None or len(self._trackers) != total_clients:
            raise ValueError("Number of clients changed after EF21 initialization")

        return flat_messages, num_honest, total_clients

    def _selection_size(
        self, configured: Optional[int], num_clients: int
    ) -> int:
        honest_lower_bound = num_clients - self.byzantine_bound
        size = (
            math.ceil(honest_lower_bound / 2)
            if configured is None
            else int(configured)
        )
        if not 1 <= size <= honest_lower_bound:
            raise ValueError(
                f"Multi-Krum selection size must be in [1, {honest_lower_bound}], got {size}"
            )
        return size

    def _multi_krum_indices(
        self, messages: torch.Tensor, selection_size: int
    ) -> torch.Tensor:
        num_clients = messages.shape[0]
        neighbor_count = num_clients - self.byzantine_bound - 2
        distances = torch.cdist(messages, messages, p=2).square()
        distances.fill_diagonal_(float("inf"))
        nearest = torch.topk(
            distances, neighbor_count, dim=1, largest=False, sorted=False
        ).values
        scores = nearest.sum(dim=1)
        # Stable sorting implements the document's lower-index tie break.
        return torch.argsort(scores, stable=True)[:selection_size]

    def _fresh_projection(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        assert self.n is not None
        projection = torch.randn(
            self.n, self.r, generator=self._rng, dtype=torch.float32
        )
        return projection.to(device=device, dtype=dtype)

    def _choose_rows(self, selected_projection_mean: torch.Tensor) -> torch.Tensor:
        scores = selected_projection_mean.square().sum(dim=1)
        top_rows = torch.argsort(scores, descending=True, stable=True)[: self.k_top]

        available = torch.ones(self.m, dtype=torch.bool, device=scores.device)
        available[top_rows] = False
        complement = torch.nonzero(available, as_tuple=False).flatten()
        permutation = torch.randperm(complement.numel(), generator=self._rng)
        random_rows = complement[permutation[: self.k_random].to(complement.device)]
        return torch.cat((top_rows, random_rows)).sort().values

    def _arc(
        self, trackers: torch.Tensor, *, inplace: bool = False
    ) -> torch.Tensor:
        total_clients = trackers.shape[0]
        arc_count = math.floor(
            2 * (self.byzantine_bound / total_clients)
            * (total_clients - self.byzantine_bound)
        )
        norms = torch.linalg.vector_norm(trackers, dim=1)
        threshold = torch.sort(norms, descending=True).values[arc_count]
        safe_norms = norms.clamp_min(torch.finfo(norms.dtype).tiny)
        scales = torch.clamp(threshold / safe_norms, max=1.0)
        if inplace:
            return trackers.mul_(scales.unsqueeze(1))
        return trackers * scales.unsqueeze(1)

    @torch.no_grad()
    def aggregate(
        self, all_gradients: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        if not all_gradients:
            return []
        flat_messages, num_honest, total_clients = self._validate_and_initialize(
            all_gradients
        )
        assert self.n is not None
        assert self.reference_shapes is not None
        assert self._trackers is not None

        projection = self._fresh_projection(
            flat_messages[0].device, flat_messages[0].dtype
        )
        projected_corrections: List[torch.Tensor] = []
        for index in range(total_clients):
            tracker = self._move_like(self._trackers[index], flat_messages[index])
            self._trackers[index] = tracker
            if index < num_honest:
                correction = flat_messages[index] - tracker
            else:
                # Byzantine inputs are arbitrary committed corrections.
                correction = flat_messages[index]
            projected_corrections.append(
                torch.matmul(correction.view(self.m, self.n), projection).div_(
                    math.sqrt(self.r)
                )
            )
        projected = torch.stack(projected_corrections)
        correction_size = self._selection_size(
            self.correction_selection_size, total_clients
        )
        correction_indices = self._multi_krum_indices(
            projected.flatten(1), correction_size
        )
        selected_projection_mean = projected[correction_indices].mean(dim=0)
        selected_rows = self._choose_rows(selected_projection_mean)
        self.last_selected_rows = selected_rows.detach()

        # q_t = q_{t-1} + Q_t r_t for every server-side tracker copy.
        for index, message in enumerate(flat_messages):
            tracker_rows = self._trackers[index].view(self.m, self.n)
            message_rows = message.view(self.m, self.n)
            if index < num_honest:
                # q + Q(u-q) copies the selected rows directly from u.
                tracker_rows[selected_rows] = message_rows[selected_rows]
            else:
                tracker_rows[selected_rows] += message_rows[selected_rows]

        tracker_matrix = torch.stack(self._trackers)
        clipped_trackers = self._arc(tracker_matrix, inplace=True)
        tracker_size = self._selection_size(
            self.tracker_selection_size, total_clients
        )
        tracker_indices = self._multi_krum_indices(
            clipped_trackers, tracker_size
        )
        aggregate = clipped_trackers[tracker_indices].mean(dim=0)
        return self._unflatten(aggregate, self.reference_shapes)

    def get_state(self) -> dict:
        return {
            "m": self.m,
            "r": self.r,
            "k_top": self.k_top,
            "k_random": self.k_random,
            "num_byzantine": self.num_byzantine,
            "byzantine_bound": self.byzantine_bound,
            "correction_selection_size": self.correction_selection_size,
            "tracker_selection_size": self.tracker_selection_size,
            "seed": self.seed,
            "d": self.d,
            "n": self.n,
            "reference_shapes": self.reference_shapes,
            "trackers": (
                None
                if self._trackers is None
                else [value.detach().cpu() for value in self._trackers]
            ),
            "last_selected_rows": (
                None
                if self.last_selected_rows is None
                else self.last_selected_rows.detach().cpu()
            ),
            "rng_state": self._rng.get_state(),
        }

    def set_state(self, state: dict) -> None:
        if not state:
            return
        for name in (
            "m",
            "r",
            "k_top",
            "k_random",
            "num_byzantine",
            "byzantine_bound",
            "correction_selection_size",
            "tracker_selection_size",
            "seed",
            "d",
            "n",
            "reference_shapes",
        ):
            if name in state:
                setattr(self, name, state[name])
        trackers = state.get("trackers")
        selected_rows = state.get("last_selected_rows")
        self._trackers = (
            None if trackers is None else [value.clone() for value in trackers]
        )
        self.last_selected_rows = (
            None if selected_rows is None else selected_rows.clone()
        )
        if state.get("rng_state") is not None:
            self._rng.set_state(state["rng_state"])
