import copy
from types import SimpleNamespace
import unittest

import torch

from fl_framework.components.optimizers.fed_dproc import FedDPRoC
from fl_framework.core.client import HonestClient
from fl_framework.core.hooks import Context
from fl_framework.core.server import Server


def _server(model, optimizer):
    return Server(
        model=model,
        aggregator=None,
        optimizer=optimizer,
        devices=["cpu"],
    )


def _clients(count=3):
    return [HonestClient(client_id=i) for i in range(count)]


def _reference_compress(optimizer, vector):
    result = torch.zeros(optimizer.k, dtype=vector.dtype, device=vector.device)
    for block in range(optimizer.p):
        idx = optimizer.hash_indices[block].to(vector.device)
        sign = optimizer.signs[block].to(vector.device)
        block_out = torch.zeros(
            optimizer.s, dtype=vector.dtype, device=vector.device
        )
        block_out.scatter_add_(0, idx, sign * vector)
        start = block * optimizer.s
        result[start : start + optimizer.s] = block_out
    return result.div_(optimizer.p**0.5)


class _FixedSampler:
    batch_size = 2

    def __init__(self, data, target):
        self.data = data
        self.target = target

    def get_sample(self):
        return torch.arange(len(self.target)), self.data, self.target


class FedDPRoCOptimizationTests(unittest.TestCase):
    def test_fed_dproc_clients_share_one_model_per_device(self):
        optimizer = FedDPRoC(lr=0.1, alpha=2, p=2, noise_multiplier=0)
        server = _server(torch.nn.Linear(4, 2), optimizer)
        clients = _clients()

        server.register_clients(clients)

        self.assertTrue(all(client.model is server.model for client in clients))
        with torch.no_grad():
            next(server.model.parameters()).add_(1)
        server.distribute_model()
        self.assertTrue(all(client.model is server.model for client in clients))

    def test_other_optimizers_keep_independent_client_models(self):
        optimizer = SimpleNamespace()
        server = _server(torch.nn.Linear(4, 2), optimizer)
        clients = _clients()

        server.register_clients(clients)

        self.assertTrue(all(client.model is not server.model for client in clients))
        self.assertEqual(
            len({id(client.model) for client in clients}), len(clients)
        )

    def test_shared_model_rejects_batch_norm(self):
        optimizer = FedDPRoC(lr=0.1, alpha=2, p=2, noise_multiplier=0)
        server = _server(torch.nn.Sequential(torch.nn.BatchNorm1d(4)), optimizer)

        with self.assertRaisesRegex(ValueError, "do not support BatchNorm"):
            server.register_clients(_clients(1))

    def test_fed_dproc_model_sharing_can_be_disabled(self):
        optimizer = FedDPRoC(
            lr=0.1,
            alpha=2,
            p=2,
            noise_multiplier=0,
            share_client_models=False,
        )
        server = _server(torch.nn.Sequential(torch.nn.BatchNorm1d(4)), optimizer)
        clients = _clients(2)

        server.register_clients(clients)

        self.assertTrue(all(client.model is not server.model for client in clients))

    def test_shared_and_independent_models_produce_identical_gradients(self):
        torch.manual_seed(11)
        base_model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.GroupNorm(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
        )
        shared_server = _server(
            copy.deepcopy(base_model),
            FedDPRoC(lr=0.1, alpha=2, p=2, noise_multiplier=0),
        )
        independent_server = _server(copy.deepcopy(base_model), SimpleNamespace())
        shared_clients = _clients(2)
        independent_clients = _clients(2)
        samples = [
            (torch.randn(2, 4), torch.tensor([0, 1])),
            (torch.randn(2, 4), torch.tensor([1, 1])),
        ]
        for clients in (shared_clients, independent_clients):
            for client, (data, target) in zip(clients, samples):
                client.sampler = _FixedSampler(data, target)

        shared_server.register_clients(shared_clients)
        independent_server.register_clients(independent_clients)

        shared_context = Context()
        independent_context = Context()
        shared_gradients = [
            client.compute_gradients(shared_context)
            for client in shared_clients
        ]
        independent_gradients = [
            client.compute_gradients(independent_context)
            for client in independent_clients
        ]
        for shared_grad, independent_grad in zip(
            shared_gradients, independent_gradients
        ):
            for actual, expected in zip(shared_grad, independent_grad):
                self.assertTrue(torch.equal(actual, expected))

    def test_count_sketch_cache_preserves_projection_and_is_reused(self):
        optimizer = FedDPRoC(
            lr=0.1, alpha=2, p=2, noise_multiplier=0, seed=7
        )
        vector = torch.arange(12, dtype=torch.float32)
        optimizer._lazy_init([vector], vector.device)

        expected = _reference_compress(optimizer, vector)
        actual_first = optimizer._compress(vector)
        cached_first = optimizer._device_hash_tables["cpu"]
        actual_second = optimizer._compress(vector)
        cached_second = optimizer._device_hash_tables["cpu"]

        self.assertTrue(torch.equal(actual_first, expected))
        self.assertTrue(torch.equal(actual_second, expected))
        self.assertIs(cached_second, cached_first)
        self.assertTrue(
            all(
                after is before
                for after, before in zip(cached_second[0], cached_first[0])
            )
        )

    def test_loading_state_invalidates_device_cache(self):
        optimizer = FedDPRoC(lr=0.1, alpha=2, p=2, noise_multiplier=0)
        vector = torch.arange(12, dtype=torch.float32)
        optimizer._lazy_init([vector], vector.device)
        optimizer._compress(vector)
        state = optimizer.get_state()

        self.assertTrue(optimizer._device_hash_tables)
        optimizer.set_state(state)
        self.assertEqual(optimizer._device_hash_tables, {})
        self.assertTrue(
            torch.equal(
                optimizer._compress(vector),
                _reference_compress(optimizer, vector),
            )
        )


if __name__ == "__main__":
    unittest.main()
