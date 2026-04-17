from typing import Tuple

import torch


def binary_packer(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.int8:
        raise TypeError("Input tensor must be int8.")

    n_rows, n_cols = tensor.shape
    words_per_row = (n_cols + 31) // 32
    pad = words_per_row * 32 - n_cols
    device = tensor.device

    if pad:
        pad_tensor = torch.ones((n_rows, pad), dtype=tensor.dtype, device=device)
        tensor = torch.cat([tensor, pad_tensor], dim=1)

    bits = ((1 - tensor) // 2).int()
    bits = bits.reshape(tensor.size(0), tensor.size(1) // 32, 32)
    weights = (2 ** torch.arange(32, dtype=torch.int32, device=device)).int()
    packed = (bits * weights).sum(dim=2, dtype=torch.int32)
    return packed.contiguous()


def binary_unpacker(
    packed_tensor: torch.Tensor, original_shape: Tuple[int, int]
) -> torch.Tensor:
    if packed_tensor.dim() != 2:
        raise ValueError(
            f"Expected a rank-2 packed tensor, got shape {tuple(packed_tensor.shape)}."
        )

    n_rows, n_cols = original_shape
    words_per_row = (n_cols + 31) // 32
    expected_shape = (n_rows, words_per_row)
    if tuple(packed_tensor.shape) != expected_shape:
        raise ValueError(
            f"Packed tensor shape {tuple(packed_tensor.shape)} does not match expected {expected_shape}."
        )

    unpacked = torch.zeros(
        n_rows,
        words_per_row * 32,
        dtype=torch.int8,
        device=packed_tensor.device,
    )
    for word_idx in range(words_per_row):
        word_data = packed_tensor[:, word_idx]
        bits = (
            word_data.unsqueeze(1) >> torch.arange(32, device=packed_tensor.device)
        ) & 1
        unpacked[:, word_idx * 32 : (word_idx + 1) * 32] = bits.to(torch.int8)

    unpacked = unpacked[:, :n_cols]
    return (1 - 2 * unpacked).to(torch.int8)
