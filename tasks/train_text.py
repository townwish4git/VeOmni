# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Text-training entry with synthetic forward/backward enabled for Muon experiments."""

import logging
import os
from types import MethodType
from typing import Any

import torch

from veomni.arguments import parse_args
from veomni.trainer.text_trainer import TextTrainer, VeOmniArguments


LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def dummy_forward_backward_step(
    self: Any, micro_batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Populate already-sharded gradients without entering the model or autograd.

    Gradients represent synthetic, already-reduced inputs to the real optimizer.
    Each micro-batch contributes its local share of the step's valid tokens so
    gradient accumulation does not scale with the number of micro-batches.
    Replica peers generate identical local samples without gradient collectives.
    """
    del micro_batch
    total_tokens = self.micro_batches_token_len["foundation_tokens"].item()
    tokens = self.micro_batch_token_len["foundation_tokens"].item()
    weight = tokens / total_tokens if total_tokens > 0 else 0.0
    generators = {}
    seed = (self.args.train.seed + self.state.global_step * 1009) % (2**63)
    for parameter in self.model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.is_meta:
            raise RuntimeError("Dummy forward/backward requires materialized model parameters, found a meta parameter")
        # Factory dispatch preserves the parameter's DTensor mesh and placements;
        # local writes avoid distributed RNG and redundant replica communication.
        gradient = torch.empty_like(parameter, requires_grad=False)
        local = gradient.to_local() if hasattr(gradient, "to_local") else gradient
        if local.device not in generators:
            generators[local.device] = torch.Generator(device=local.device).manual_seed(seed)
        local.normal_(mean=0.0, std=0.01, generator=generators[local.device]).mul_(weight)
        if parameter.grad is None:
            parameter.grad = gradient
        else:
            accumulated = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
            accumulated.add_(local)
    # The training loop only logs these scalars; CPU tensors avoid a device sync.
    loss = torch.tensor(weight, dtype=torch.float32)
    return loss, {"dummy_loss": loss}, {}


def enable_dummy_forward_backward(trainer: TextTrainer) -> None:
    """Replace only this TextTrainer instance's forward/backward implementation."""
    trainer.base.forward_backward_step = MethodType(dummy_forward_backward_step, trainer.base)
    LOGGER.warning(
        "DUMMY FORWARD/BACKWARD ENABLED: no model forward or autograd backward; "
        "synthetic local gradients and dummy_loss only. Data loading, gradient clipping, "
        "optimizer/scheduler updates and configured callbacks/checkpoints remain active."
    )


if __name__ == "__main__":
    dummy_mode = os.environ.get("VEOMNI_DUMMY_FWD_BWD", "1")
    if dummy_mode not in ("0", "1"):
        raise ValueError("VEOMNI_DUMMY_FWD_BWD must be 0 or 1")
    args = parse_args(VeOmniArguments)
    trainer = TextTrainer(args)
    if dummy_mode == "1":
        enable_dummy_forward_backward(trainer)
    trainer.train()
