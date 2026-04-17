"""LittleBit QAT helpers for DFlash draft models."""

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class LittleBitDFlashLosses:
    loss: torch.Tensor
    kd_loss: torch.Tensor
    l2l_loss: torch.Tensor


def kd_loss_like_littlebit(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    student_log_prob = F.log_softmax(student_logits, dim=-1)
    teacher_prob = F.softmax(teacher_logits, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")


def mse_loss_like_littlebit(
    student_hidden_states: Iterable[torch.Tensor],
    teacher_hidden_states: Iterable[torch.Tensor],
    *,
    l2l_loss_scale: float,
) -> torch.Tensor:
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    loss = None
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        for student_hidden, teacher_hidden in zip(
            student_hidden_states, teacher_hidden_states
        ):
            current = F.mse_loss(student_hidden, teacher_hidden)
            loss = current if loss is None else loss + current

    if loss is None:
        sample = next(iter(student_hidden_states), None)
        if sample is None:
            raise ValueError("student_hidden_states must not be empty.")
        loss = torch.zeros((), device=sample.device, dtype=sample.dtype)
    return l2l_loss_scale * loss


def compute_littlebit_dflash_losses(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_hidden_states: Iterable[torch.Tensor],
    teacher_hidden_states: Iterable[torch.Tensor],
    *,
    l2l_loss_scale: float,
    kd_loss_scale: float = 1.0,
) -> LittleBitDFlashLosses:
    kd_loss = kd_loss_scale * kd_loss_like_littlebit(student_logits, teacher_logits)
    l2l_loss = mse_loss_like_littlebit(
        student_hidden_states,
        teacher_hidden_states,
        l2l_loss_scale=l2l_loss_scale,
    )
    return LittleBitDFlashLosses(
        loss=kd_loss + l2l_loss,
        kd_loss=kd_loss,
        l2l_loss=l2l_loss,
    )
