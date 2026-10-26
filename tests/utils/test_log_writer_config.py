# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration serialization without connecting to external logging services."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rsl_rl.utils import wandb_log_writer
from rsl_rl.utils.neptune_log_writer import NeptuneLogWriter
from rsl_rl.utils.wandb_log_writer import WandbLogWriter


@dataclass
class DataclassConfig:
    """Representative dataclass-based simulator configuration."""

    seed: int = 7


class ObjectConfig:
    """Representative Isaac Lab configuration object."""

    def to_dict(self) -> dict:
        """Return its serializable fields."""
        return {"seed": 7}


@pytest.mark.parametrize("env_cfg", [{}, {"seed": 7}, DataclassConfig(), ObjectConfig()])
@pytest.mark.parametrize("writer_type", [WandbLogWriter, NeptuneLogWriter])
def test_external_writers_accept_environment_configurations(
    monkeypatch: pytest.MonkeyPatch, env_cfg: object, writer_type: type
) -> None:
    """Generic dictionaries and established simulator config types serialize unchanged."""
    backend = SimpleNamespace(config={})
    monkeypatch.setattr(wandb_log_writer, "wandb", backend)
    writer = object.__new__(writer_type)
    writer.run = {}
    writer.store_config(env_cfg, {"algorithm": {}})
    stored = backend.config if writer_type is WandbLogWriter else writer.run
    assert stored == {"env_cfg": env_cfg if isinstance(env_cfg, dict) else {"seed": 7}, "train_cfg": {"algorithm": {}}}
