import torch

from specforge.core.littlebit_dflash import (
    compute_littlebit_dflash_losses,
    compute_littlebit_dflash_losses_from_hidden,
)
from specforge.littlebit.packing import binary_packer, binary_unpacker


def test_binary_pack_roundtrip():
    tensor = torch.tensor([[1, -1, 1, -1, 1, -1, 1, -1]], dtype=torch.int8)
    packed = binary_packer(tensor)
    unpacked = binary_unpacker(packed, tensor.shape)
    assert torch.equal(unpacked, tensor)


def test_compute_littlebit_dflash_losses():
    student_logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    teacher_logits = torch.tensor([[[0.5, 0.5], [0.2, 0.8]]], dtype=torch.float32)
    student_hidden = (
        torch.tensor([[[1.0, 2.0]]], dtype=torch.float32),
        torch.tensor([[[3.0, 4.0]]], dtype=torch.float32),
    )
    teacher_hidden = (
        torch.tensor([[[1.5, 1.5]]], dtype=torch.float32),
        torch.tensor([[[2.5, 4.5]]], dtype=torch.float32),
    )

    losses = compute_littlebit_dflash_losses(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_hidden_states=student_hidden,
        teacher_hidden_states=teacher_hidden,
        l2l_loss_scale=2.0,
        kd_loss_scale=1.0,
    )

    assert losses.loss.item() > 0
    assert losses.kd_loss.item() > 0
    assert losses.l2l_loss.item() > 0
    assert torch.isclose(losses.loss, losses.kd_loss + losses.l2l_loss)


def test_chunked_hidden_losses_match_full_logits():
    lm_head = torch.nn.Linear(3, 3, bias=False)
    lm_head.weight.data.copy_(torch.eye(3))
    lm_head.requires_grad_(False)

    student_output_hidden = torch.randn(1, 5, 3, dtype=torch.float32)
    teacher_output_hidden = torch.randn(1, 5, 3, dtype=torch.float32)
    student_hidden = (torch.randn(1, 5, 3, dtype=torch.float32),)
    teacher_hidden = (torch.randn(1, 5, 3, dtype=torch.float32),)

    full = compute_littlebit_dflash_losses(
        student_logits=lm_head(student_output_hidden),
        teacher_logits=lm_head(teacher_output_hidden),
        student_hidden_states=student_hidden,
        teacher_hidden_states=teacher_hidden,
        l2l_loss_scale=2.0,
        kd_loss_scale=1.0,
    )
    chunked = compute_littlebit_dflash_losses_from_hidden(
        student_output_hidden=student_output_hidden,
        teacher_output_hidden=teacher_output_hidden,
        student_hidden_states=student_hidden,
        teacher_hidden_states=teacher_hidden,
        lm_head=lm_head,
        l2l_loss_scale=2.0,
        kd_loss_scale=1.0,
        logit_chunk_size=2,
    )

    assert torch.allclose(chunked.loss, full.loss, atol=1e-6)
    assert torch.allclose(chunked.kd_loss, full.kd_loss, atol=1e-6)
    assert torch.allclose(chunked.l2l_loss, full.l2l_loss, atol=1e-6)
