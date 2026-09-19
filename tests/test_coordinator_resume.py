import os
import tempfile
import unittest

import torch

from fl_framework.core.coordinator import Coordinator


class _Optimizer:
    def register_hooks(self):
        pass


class _Server:
    def __init__(self):
        self.optimizer = _Optimizer()
        self.aggregator = None
        self.loaded_state = None
        self.distribute_calls = 0
        self.test_calls = 0

    def register_clients(self, clients):
        self.clients = clients

    def set_state(self, state):
        self.loaded_state = state

    def distribute_model(self):
        self.distribute_calls += 1

    def test(self):
        self.test_calls += 1
        return 0.0, {}


class CoordinatorResumeTests(unittest.TestCase):
    def test_resume_continues_after_saved_round_without_duplicate_test(self):
        server = _Server()
        with tempfile.TemporaryDirectory() as save_path:
            checkpoint = os.path.join(save_path, "state_round_2.pth")
            torch.save(
                {
                    "current_round": 2,
                    "history": {
                        "loss": [0.0, 1.0, 2.0, 3.0],
                        "metrics": [{}, {}, {}, {}],
                    },
                    "server_state": {"model": "restored"},
                },
                checkpoint,
            )
            coordinator = Coordinator(
                server=server,
                clients=[],
                num_byzantine=0,
                num_rounds=5,
                num_round_steps=0,
                save_path=save_path,
            )
            visited_rounds = []
            coordinator._run_round = lambda: visited_rounds.append(
                coordinator.context.current_round
            )
            coordinator.save_state = lambda filepath: None

            coordinator.run(resume_from=checkpoint)

        self.assertEqual(visited_rounds, [3, 4])
        self.assertEqual(server.loaded_state, {"model": "restored"})
        self.assertEqual(server.distribute_calls, 1)
        self.assertEqual(server.test_calls, 0)
        self.assertEqual(len(coordinator.history["loss"]), 4)


if __name__ == "__main__":
    unittest.main()
