# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for SAC and the standard replay interaction contract."""

import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms import SAC
from rsl_rl.runners import OffPolicyRunner, OnPolicyRunner
from tests.runners.test_on_policy_runner import DummyEnv


def _cfg() -> dict:
    return {
        "num_steps_per_env": 3,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [16, 16],
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "SquashedGaussianDistribution"},
        },
        "critic": {"class_name": "QMLPModel", "hidden_dims": [16, 16], "obs_normalization": True},
        "algorithm": {"class_name": "SAC", "batch_size": 8, "learning_starts": 0, "replay_buffer_size": 40},
    }


def _algorithm() -> SAC:
    cfg = _cfg()
    cfg["multi_gpu"] = None
    env = DummyEnv()
    return SAC.construct_algorithm(env.get_observations(), env, cfg, "cpu")




@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for replay parity"),
        ),
    ],
)
@pytest.mark.parametrize("scenario", ["boundaries", "wrapped", "independent"])
def test_n_step_targets_match_legacy_dense_goldens(device: str, scenario: str) -> None:
    """SAC retains targets captured from the actual e9adc75 dense replay, including partial windows."""
    capacity, n_steps, rewards, ends, terminals, expected = {
        "boundaries": (
            10,
            3,
            [1, 2, 3, 4, 5, 6],
            [2, 4],
            [4],
            {
                0: (5.229999542236328, 102, 0.7289999127388),
                1: (4.699999809265137, 102, 0.809999942779541),
                2: (3, 102, 0.8999999761581421),
                3: (8.5, 104, 0),
                4: (5, 104, 0),
                5: (6, 105, 0.8999999761581421),
            },
        ),
        "wrapped": (
            4,
            2,
            [1, 2, 3, 4, 5, 6],
            [],
            [],
            {
                2: (6.599999904632568, 103, 0.809999942779541),
                3: (8.5, 104, 0.809999942779541),
                4: (10.399999618530273, 105, 0.809999942779541),
                5: (6, 105, 0.8999999761581421),
            },
        ),
        "independent": (
            8,
            2,
            [1, 1, 1],
            [0],
            [0],
            {
                0: (1, 100, 0),
                1: (1.899999976158142, 102, 0.809999942779541),
                2: (1, 102, 0.8999999761581421),
                10: (1.899999976158142, 111, 0.809999942779541),
                11: (1.899999976158142, 112, 0.809999942779541),
                12: (1, 112, 0.8999999761581421),
            },
        ),
    }[scenario]
    env = DummyEnv(device=device)
    env.num_envs = 2 if scenario == "independent" else 1
    cfg = _cfg()
    cfg["multi_gpu"] = None
    cfg["algorithm"].update(replay_buffer_size=capacity, n_steps=n_steps, gamma=0.9, batch_size=256)
    alg = SAC.construct_algorithm(env.get_observations(), env, cfg, device)
    for step, reward in enumerate(rewards):
        start = step + torch.arange(env.num_envs, device=device, dtype=torch.float) * 10
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        terminated = torch.zeros_like(done)
        done[0], terminated[0] = step in ends, step in terminals
        alg.storage.add(
            TensorDict(
                {
                    "observations": {"policy": start[:, None].expand(-1, 8)},
                    "actions": torch.zeros(env.num_envs, 4, device=device),
                    "rewards": torch.full((env.num_envs,), float(reward), device=device),
                    "next_observations": {"policy": (start + 100)[:, None].expand(-1, 8)},
                    "dones": done,
                    "terminated": terminated,
                },
                batch_size=[env.num_envs],
                device=device,
            )
        )
    torch.manual_seed(0)
    batch = alg._sample_batch()
    rows = (
        torch.stack(
            (
                batch.observations["policy"][:, 0],
                batch.rewards,
                batch.next_observations["policy"][:, 0],
                batch.bootstrap_discounts,
            ),
            dim=1,
        )
        .cpu()
        .tolist()
    )
    assert {int(row[0]) for row in rows} == expected.keys()
    for start, *values in rows:
        assert values == pytest.approx(expected[int(start)], abs=1e-6)


def test_terminal_observations_and_missing_final() -> None:
    """All done successors use final observations; only exclusive timeouts bootstrap."""
    alg = _algorithm()
    obs = DummyEnv().get_observations()
    dones = torch.tensor([True, True, False, False])
    alg.act(obs)
    with pytest.raises(ValueError, match="final_obs"):
        alg.process_env_step(obs, torch.ones(4), dones, {})
    alg.process_env_step(
        obs,
        torch.ones(4),
        dones,
        {
            "time_outs": torch.tensor([False, True, False, False]),
            "final_obs": TensorDict({"policy": torch.full_like(obs["policy"], 9.0)}, batch_size=[4]),
        },
    )
    torch.testing.assert_close(alg.storage.data["next_observations", "policy"][0, :2], torch.full((2, 8), 9.0))
    assert alg.storage.data["terminated"][0].tolist() == [True, False, False, False]
    assert alg.actor.obs_normalizer.count == 0
    alg.update()
    assert alg.actor.obs_normalizer.count == 4


def test_reward_relabeler_and_partial_n_step() -> None:
    """Reward extension points are evaluated at sampling time, including a newest-step cut."""

    class Relabeler(SAC):
        def _make_transition(self, *args: object, **kwargs: object) -> TensorDict:
            step = super()._make_transition(*args, **kwargs)
            step["bonus"] = torch.full_like(step["rewards"], 2.0)
            return step

        def _window_rewards(self, window: dict[str | tuple[str, ...], torch.Tensor]) -> torch.Tensor:
            return window["rewards"] + window["bonus"]

    cfg = _cfg()
    cfg["multi_gpu"] = None
    cfg["algorithm"].update(class_name=Relabeler, n_steps=3)
    env = DummyEnv()
    alg = Relabeler.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    obs = env.get_observations()
    alg.act(obs)
    alg.process_env_step(obs, torch.ones(4), torch.zeros(4), {})
    batch = alg._sample_batch()
    torch.testing.assert_close(batch.rewards, torch.full((8,), 3.0))
    torch.testing.assert_close(batch.bootstrap_discounts, torch.full((8,), alg.gamma))


@pytest.mark.parametrize("network", [None, "ResidualMLP"])
def test_opt_in_flash_components_and_export(network: str | None, tmp_path: Path) -> None:
    """Categorical updates, return scaling, held noise and residual export work through the standard runner."""
    cfg = _cfg()
    cfg["algorithm"].update(
        critic_loss="categorical",
        num_bins=11,
        target_entropy_std=0.15,
        reward_normalization=True,
        exploration="held",
        learning_rate_schedule="cosine",
        schedule_updates=10,
        num_updates=2,
    )
    if network:
        for role in ("actor", "critic"):
            cfg[role]["network_cfg"] = {"class_name": network, "hidden_dim": 16, "num_blocks": 1, "expansion": 2}
    runner = OffPolicyRunner(DummyEnv(), cfg)
    runner.learn(3)
    assert runner.alg.num_critic_updates == 6
    assert runner.alg._return_normalizer.count == 36
    sample = runner.env.get_observations()
    runner.alg.eval_mode()
    expected = runner.alg.get_policy()(sample)
    runner.export_policy_to_jit(str(tmp_path))
    scripted = torch.jit.load(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(scripted(sample["policy"]), expected)
    checkpoint = runner.alg.save()
    restored = OffPolicyRunner(DummyEnv(), cfg)
    restored.alg.load(checkpoint, None, True)
    assert restored.alg._schedulers[0].last_epoch == runner.alg._schedulers[0].last_epoch


def _mirror(env: object, obs: TensorDict | None, actions: torch.Tensor | None) -> tuple:
    """Mirror all inputs around zero while retaining the original batch first."""
    return (
        torch.cat([obs, -obs]) if obs is not None else None,
        torch.cat([actions, -actions]) if actions is not None else None,
    )


@pytest.mark.parametrize("augmentation", [False, True])
def test_symmetry_replay_endpoints_and_actor_loss(augmentation: bool) -> None:
    """Symmetry handles replay endpoint pairs and the existing auxiliary actor loss."""
    cfg = _cfg()
    cfg["algorithm"]["symmetry_cfg"] = {
        "data_augmentation_func": _mirror,
        "use_data_augmentation": augmentation,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.1,
    }
    runner = OffPolicyRunner(DummyEnv(), cfg)
    runner.learn(2)
    batch = runner.alg._augment_batch(runner.alg._sample_batch())
    if augmentation:
        n = batch.original_batch_size
        torch.testing.assert_close(batch.observations["policy"][n:], -batch.observations["policy"][:n])
        torch.testing.assert_close(batch.next_observations["policy"][n:], -batch.next_observations["policy"][:n])
        torch.testing.assert_close(batch.actions[n:], -batch.actions[:n])
    assert torch.isfinite(torch.tensor(runner.alg.update()["symmetry"]))


