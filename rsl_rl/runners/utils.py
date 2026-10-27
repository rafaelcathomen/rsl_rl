# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint metadata and serialization for prepared policy export modules."""

from __future__ import annotations

import torch
from pathlib import Path


def save_checkpoint(path: str, state: dict, iteration: int, infos: dict | None) -> None:
    """Save an algorithm's state with the standard runner metadata."""
    state["iter"] = iteration
    state["infos"] = infos
    torch.save(state, path)


def export_jit(model: torch.nn.Module, path: str, filename: str) -> None:
    """Script a prepared deterministic policy on CPU in evaluation mode."""
    model = model.to("cpu").eval()
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    torch.jit.script(model).save(str(target / filename))


def export_onnx(model: torch.nn.Module, path: str, filename: str, verbose: bool) -> None:
    """Export a prepared policy with its declared ONNX inputs and outputs."""
    model = model.to("cpu").eval()
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        model.get_dummy_inputs(),  # type: ignore
        str(target / filename),
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=model.input_names,  # type: ignore
        output_names=model.output_names,  # type: ignore
    )
