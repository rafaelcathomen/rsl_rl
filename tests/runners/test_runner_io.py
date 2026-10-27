# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared checkpoint and export contracts for the PPO and SAC runners."""

from __future__ import annotations

import copy
import torch
from pathlib import Path

import onnx
import pytest
from onnx.reference import ReferenceEvaluator

from rsl_rl.runners import DistillationRunner, OffPolicyRunner, OnPolicyRunner
from tests.algorithms.test_sac import DummyEnv as SacEnv
from tests.algorithms.test_sac import _train_cfg as sac_cfg
from tests.runners.test_on_policy_runner import DummyEnv as PpoEnv
from tests.runners.test_on_policy_runner import _make_train_cfg as ppo_cfg


@pytest.fixture(params=["ppo", "sac"])
def runner(request: pytest.FixtureRequest) -> OnPolicyRunner | OffPolicyRunner:
    """Build a normalized policy, including BatchNorm in the SAC case."""
    if request.param == "ppo":
        cfg = ppo_cfg()
        runner_type, env = OnPolicyRunner, PpoEnv()
    else:
        cfg = sac_cfg()
        cfg["actor"]["network_cfg"] = {"class_name": "ResidualMLP", "hidden_dim": 8, "num_blocks": 1}
        runner_type, env = OffPolicyRunner, SacEnv()
    cfg["actor"]["obs_normalization"] = True
    return runner_type(env, cfg)


def test_selected_checkpoint_load(runner: OnPolicyRunner | OffPolicyRunner, tmp_path: Path) -> None:
    """Actor-only loading preserves the fresh critic, optimizer and runner iteration."""
    runner.learn(2)
    checkpoint = tmp_path / "model.pt"
    runner.save(str(checkpoint), infos={"experiment": "selected-load"})
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["iter"] == runner.current_learning_iteration

    restored = type(runner)(type(runner.env)(), runner.cfg)
    before = copy.deepcopy(restored.alg.save())
    restored.current_learning_iteration = 17
    infos = restored.load(str(checkpoint), load_cfg={"actor": True}, map_location="cpu")
    after = restored.alg.save()

    assert infos == {"experiment": "selected-load"}
    assert restored.current_learning_iteration == 17
    torch.testing.assert_close(after["actor_state_dict"], saved["actor_state_dict"])
    torch.testing.assert_close(after["critic_state_dict"], before["critic_state_dict"])
    for key in before:
        if key.endswith("optimizer_state_dict"):
            assert after[key]["state"] == {}


def test_checkpoint_strict_loading(runner: OnPolicyRunner | OffPolicyRunner, tmp_path: Path) -> None:
    """The runner forwards strict state-dict loading independently of load selection."""
    checkpoint = tmp_path / "model.pt"
    runner.save(str(checkpoint))
    state = torch.load(checkpoint, weights_only=False)
    state["actor_state_dict"]["unexpected"] = torch.zeros(1)
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        runner.load(str(checkpoint), load_cfg={"actor": True})
    assert runner.load(str(checkpoint), load_cfg={"actor": True}, strict=False) is None


def test_policy_exports(runner: OnPolicyRunner | OffPolicyRunner, tmp_path: Path) -> None:
    """JIT and ONNX match normalized deterministic inference without switching the live policy's mode."""
    runner.learn(2)
    target = tmp_path / "exports"
    runner.export_policy_to_jit(str(target), "actor.pt")
    runner.export_policy_to_onnx(str(target), "actor.onnx")
    assert runner.alg.get_policy().training

    obs = runner.env.get_observations()[:1]
    expected = runner.get_inference_policy()(obs).detach()
    exported = torch.jit.load(str(target / "actor.pt"))
    torch.testing.assert_close(exported(obs["policy"]), expected)

    model = onnx.load(target / "actor.onnx")
    onnx.checker.check_model(model)
    actions = ReferenceEvaluator(model).run(None, {"obs": obs["policy"].numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(actions), expected)


def test_off_policy_rejects_distributed_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported distributed launches fail before touching the simulator or initializing NCCL."""
    monkeypatch.setenv("WORLD_SIZE", "2")

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Distributed SAC must fail before initialization")

    env = SacEnv()
    monkeypatch.setattr(env, "get_observations", unexpected)
    monkeypatch.setattr(torch.distributed, "init_process_group", unexpected)
    with pytest.raises(ValueError, match="one training process"):
        OffPolicyRunner(env, sac_cfg(), device="cuda:0")


def test_distillation_checkpoint_and_export(tmp_path: Path) -> None:
    """A PPO checkpoint seeds a teacher, while distillation checkpoints restore student training."""
    cfg = ppo_cfg()
    teacher = OnPolicyRunner(PpoEnv(), copy.deepcopy(cfg))
    teacher.save(str(tmp_path / "teacher.pt"))
    cfg["algorithm"] = {"class_name": "Distillation", "gradient_length": 4}
    cfg["obs_groups"] = {"student": ["policy"], "teacher": ["policy"]}
    cfg["student"] = cfg.pop("actor")
    cfg["teacher"] = copy.deepcopy(cfg["student"])
    cfg.pop("critic")
    runner = DistillationRunner(PpoEnv(), cfg)
    runner.load(str(tmp_path / "teacher.pt"))
    assert runner.current_learning_iteration == 0
    before = copy.deepcopy(runner.alg.student.state_dict())
    runner.learn(2)
    assert any(not torch.equal(before[key], value) for key, value in runner.alg.student.state_dict().items())

    runner.save(str(tmp_path / "student.pt"))
    restored = DistillationRunner(PpoEnv(), cfg)
    restored.load(str(tmp_path / "student.pt"))
    assert restored.alg.teacher_loaded
    assert restored.current_learning_iteration == runner.current_learning_iteration
    torch.testing.assert_close(restored.alg.save(), runner.alg.save())
    restored.export_policy_to_jit(str(tmp_path))
    obs = restored.env.get_observations()
    exported = torch.jit.load(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(exported(obs["policy"]), restored.get_inference_policy()(obs))
