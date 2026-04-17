import torch

from specforge.core.littlebit_dflash import compute_littlebit_dflash_losses
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
