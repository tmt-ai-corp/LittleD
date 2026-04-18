import torch
from argparse import Namespace

from specforge.core.littlebit_dflash import (
    compute_littlebit_dflash_losses,
    compute_littlebit_dflash_losses_from_hidden,
)
from specforge.modeling.draft.dflash import (
    build_ddtree_tree,
    compile_ddtree_tree,
    find_first_stop_sequence,
    follow_verified_tree,
)
from specforge.littlebit import apply_littlebit_patch
from specforge.littlebit.packing import binary_packer, binary_unpacker
from specforge.littlebit.utils import _load_state_dict_allow_meta


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


def test_littlebit_meta_load_assigns_real_tensors():
    quant_args = Namespace(
        quant_func="STEBinary",
        split_dim=8,
        eff_bit=None,
        residual=False,
        kv_factor=1.0,
        min_split_dim=8,
    )
    source = torch.nn.Sequential(torch.nn.Linear(16, 12, bias=False))
    source = apply_littlebit_patch(source, quant_args, do_train=True)

    target = torch.nn.Sequential(torch.nn.Linear(16, 12, bias=False))
    target = apply_littlebit_patch(target, quant_args, do_train=False)
    assert any(param.is_meta for param in target.parameters())

    state_dict = source.state_dict()
    state_dict["0.U"] = source[0].U.detach()
    state_dict["0.V"] = source[0].V.detach()
    state_dict.pop("0.U_packed")
    state_dict.pop("0.U_shape")
    state_dict.pop("0.V_packed")
    state_dict.pop("0.V_shape")

    missing, unexpected = _load_state_dict_allow_meta(target, state_dict, strict=False)

    assert not missing
    assert not unexpected
    assert not any(param.is_meta for param in target.parameters())
    assert torch.allclose(target[0].U, source[0].U)


def test_find_first_stop_sequence():
    token_ids = torch.tensor([10, 11, 12, 13, 11, 12])

    assert find_first_stop_sequence(token_ids, [[11, 12]]) == 1
    assert find_first_stop_sequence(token_ids, [[13], [11, 12]]) == 1
    assert find_first_stop_sequence(token_ids, [[99]]) is None


def test_build_ddtree_tree_and_follow_verified_path():
    logits = torch.full((3, 8), -10.0)
    logits[0, 1] = 4.0
    logits[0, 2] = 3.0
    logits[1, 3] = 4.0
    logits[1, 4] = 3.0
    logits[2, 5] = 4.0

    node_token_ids, node_depths, parents, child_maps, visibility = build_ddtree_tree(
        logits,
        budget=4,
    )

    assert node_token_ids.tolist() == [1, 3, 5, 2]
    assert node_depths.tolist() == [1, 2, 3, 1]
    assert parents == [-1, 0, 1, 2, 0]
    assert visibility.shape == (5, 5)
    assert visibility[3, [0, 1, 2, 3]].all()
    assert not visibility[3, 4]

    accepted_indices, next_token = follow_verified_tree(
        child_maps,
        torch.tensor([[1, 3, 5, 0, 0]]),
    )

    assert accepted_indices == [0, 1, 2, 3]
    assert next_token == 0


def test_compile_ddtree_tree_builds_ancestor_mask():
    verify_input_ids_buffer = torch.empty((1, 5), dtype=torch.long)
    verify_position_ids_buffer = torch.empty((1, 5), dtype=torch.long)
    attention_mask_buffer = torch.zeros((1, 1, 5, 16), dtype=torch.float32)
    tree_visibility_buffer = torch.empty((5, 5), dtype=torch.bool)
    visibility = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, False, True],
        ],
        dtype=torch.bool,
    )

    input_ids, position_ids, attention_mask, tree_start, tree_length = (
        compile_ddtree_tree(
            root_token_id=torch.tensor(10),
            start=7,
            node_token_ids=torch.tensor([11, 12]),
            node_depths=torch.tensor([1, 1]),
            visibility_cpu=visibility,
            past_length=7,
            dtype=torch.float32,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=0,
            previous_tree_length=0,
        )
    )

    assert input_ids.tolist() == [[10, 11, 12]]
    assert position_ids.tolist() == [[7, 8, 8]]
    assert tree_start == 7
    assert tree_length == 3
    assert attention_mask.shape == (1, 1, 3, 10)
    assert attention_mask[0, 0, 1, 7] == 0
    assert attention_mask[0, 0, 1, 8] == 0
    assert attention_mask[0, 0, 1, 9] < -1e20
