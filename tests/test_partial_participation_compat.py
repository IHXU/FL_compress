import collections
import unittest

import torch

from fl_framework.components.aggregators.krum import KrumAggregator
from fl_framework.components.aggregators.mean import MeanAggregator
from fl_framework.components.optimizers.d_byz_sgdm import DByzSGDM
from fl_framework.components.optimizers.sgd import SGD
from fl_framework.core.coordinator import Coordinator
from fl_framework.core.hooks import HookType, hook_registry
from fl_framework.core.server import Server


class _CountingClient:
    client_type = "Honest"

    def __init__(self, client_id):
        self.client_id = client_id
        self.model = None
        self.device = "cpu"
        self.calls = 0
        self.model_seen_during_compute = None

    def compute_gradients(self, context):
        self.calls += 1
        self.model_seen_during_compute = [
            p.detach().clone() for p in self.model.parameters()
        ]
        return [torch.ones_like(p) for p in self.model.parameters()]


class PartialParticipationCompatibilityTests(unittest.TestCase):
    def setUp(self):
        # The hook registry is process-global. Give each test an isolated
        # registry without disturbing hooks owned by other test modules.
        self._saved_hooks = hook_registry._hooks
        hook_registry._hooks = collections.defaultdict(list)

    def tearDown(self):
        hook_registry._hooks = self._saved_hooks

    @staticmethod
    def _clients(count):
        return [_CountingClient(i) for i in range(count)]

    def test_optimizer_without_mask_keeps_full_client_workflow(self):
        optimizer = SGD(lr=0.1, weight_decay=0.0)
        server = Server(
            torch.nn.Linear(2, 1, bias=False),
            MeanAggregator(),
            optimizer,
            devices=["cpu"],
        )
        clients = self._clients(3)
        coordinator = Coordinator(server, clients, 0, 1, 1)

        coordinator.context.current_round = 0
        coordinator.context.current_step = 0
        coordinator._run_step()

        self.assertEqual([client.calls for client in clients], [1, 1, 1])
        for client in clients:
            for server_param, client_param in zip(
                server.model.parameters(), client.model.parameters()
            ):
                self.assertTrue(torch.equal(server_param, client_param))

    def test_d_byz_mask_skips_only_inactive_clients_and_syncs_active_model(self):
        optimizer = DByzSGDM(lr=0.1, momentum=0.9, participation=0.1)
        server = Server(
            torch.nn.Linear(1, 1, bias=False),
            KrumAggregator(0),
            optimizer,
            devices=["cpu"],
        )
        clients = self._clients(3)
        coordinator = Coordinator(server, clients, 0, 1, 1)

        def fixed_mask(context):
            mask = [False, True, False]
            context.extra["d_byz_sgdm_participants"] = mask
            context.extra["active_client_mask"] = mask

        # Registered after D-Byz's sampler, so this deterministic mask wins.
        hook_registry.register(HookType.BEFORE_STEP_SERVER, fixed_mask)
        with torch.no_grad():
            next(server.model.parameters()).fill_(7.0)

        coordinator.context.current_round = 0
        coordinator.context.current_step = 0
        coordinator._run_step()

        self.assertEqual([client.calls for client in clients], [0, 1, 0])
        self.assertEqual(clients[1].model_seen_during_compute[0].item(), 7.0)
        self.assertTrue(all(buf is not None for buf in optimizer._momentum_cache))


if __name__ == "__main__":
    unittest.main()
