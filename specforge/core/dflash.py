# coding=utf-8
"""DFlash training helpers and wrapper modules."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.dflash import DFlashDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None


def create_dflash_sdpa_mask(anchor_positions, block_keep_mask, S, block_size, device):
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)  # (1, 1, Q_LEN, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(
        1, 1, 1, -1
    )  # (1, 1, 1, KV_LEN)

    q_block_ids = q_indices // block_size

    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    final_mask = (mask_context | mask_draft) & valid_block
    return final_mask


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
):
    """Construct Flex Attention BlockMask for DFlash training.

    KV: [Context (S tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos).
      2. Intra-block attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    return create_block_mask(
        dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


@dataclass
class DFlashTrainingView:
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    noise_embedding: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_ids: torch.Tensor
    weight_mask: torch.Tensor


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with block-wise CE loss."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        fixed_num_anchors: bool = False,
        loss_decay_gamma: Optional[float] = None,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.fixed_num_anchors = fixed_num_anchors
        self.loss_decay_gamma = loss_decay_gamma

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_valid_count = int(valid_counts.max().item())
        max_n = (
            self.num_anchors
            if self.fixed_num_anchors
            else min(self.num_anchors, max_valid_count - 1)
        )

        if max_n <= 0 or max_valid_count <= 1:
            raise ValueError("should preprocess the data.")

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        selected = gathered[:, : min(max_n, gathered.size(1))]
        if selected.size(1) < max_n:
            pad = torch.full(
                (bsz, max_n - selected.size(1)),
                seq_len + 1,
                dtype=selected.dtype,
                device=device,
            )
            selected = torch.cat([selected, pad], dim=1)
        anchors = selected.sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def prepare_noise_input(
        self, input_ids: torch.Tensor, block_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prepare noise input: first token of each block is real, rest are MASK."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if block_ids is not None:
            is_block_start = torch.ones(bsz, seq_len, dtype=torch.bool, device=device)
            is_block_start[:, 1:] = block_ids[:, 1:] != block_ids[:, :-1]
        else:
            positions = torch.arange(seq_len, device=device)
            is_block_start = (positions % self.block_size) == 0
            is_block_start = is_block_start.unsqueeze(0).expand(bsz, -1)

        noise_input_ids = torch.full_like(input_ids, self.mask_token_id)
        noise_input_ids[is_block_start] = input_ids[is_block_start]
        return noise_input_ids

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def _create_label_tensors(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(
                -(k - 1).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        return target_ids, weight_mask

    def prepare_training_view(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTrainingView:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )

        target_ids, weight_mask = self._create_label_tensors(
            input_ids=input_ids,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )

        return DFlashTrainingView(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            noise_embedding=noise_embedding,
            position_ids=full_position_ids,
            attention_mask=dflash_attn_mask,
            target_ids=target_ids,
            weight_mask=weight_mask,
        )

    def forward_hidden_from_view(
        self,
        hidden_states: torch.Tensor,
        training_view: DFlashTrainingView,
        draft_model: Optional[DFlashDraftModel] = None,
        output_hidden_states: bool = False,
    ):
        draft_model = draft_model or self.draft_model
        model_outputs = draft_model(
            position_ids=training_view.position_ids,
            noise_embedding=training_view.noise_embedding,
            target_hidden=hidden_states,
            attention_mask=training_view.attention_mask,
            output_hidden_states=output_hidden_states,
        )

        if output_hidden_states:
            output_hidden, hidden_states_list = model_outputs
        else:
            output_hidden = model_outputs
            hidden_states_list = None

        if output_hidden_states:
            return output_hidden, hidden_states_list
        return output_hidden

    def forward_from_view(
        self,
        hidden_states: torch.Tensor,
        training_view: DFlashTrainingView,
        draft_model: Optional[DFlashDraftModel] = None,
        output_hidden_states: bool = False,
    ):
        model_outputs = self.forward_hidden_from_view(
            hidden_states=hidden_states,
            training_view=training_view,
            draft_model=draft_model,
            output_hidden_states=output_hidden_states,
        )

        if output_hidden_states:
            output_hidden, hidden_states_list = model_outputs
        else:
            output_hidden = model_outputs
            hidden_states_list = None

        logits = self.lm_head(output_hidden)
        if output_hidden_states:
            return logits, hidden_states_list
        return logits

    def compute_loss_and_accuracy_from_hidden(
        self,
        output_hidden: torch.Tensor,
        training_view: DFlashTrainingView,
        *,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_loss = output_hidden.new_zeros(())
        total_correct = output_hidden.new_zeros(())
        flat_targets = training_view.target_ids.reshape(output_hidden.shape[0], -1)
        flat_weights = training_view.weight_mask.reshape(output_hidden.shape[0], -1)
        valid_token_count = flat_weights.sum() + 1e-6
        actual_token_count = (flat_weights > 0.0).sum() + 1e-6

        seq_len = output_hidden.shape[1]
        if chunk_size is None or chunk_size <= 0:
            chunk_size = seq_len

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            logits = self.lm_head(output_hidden[:, start:end, :])
            flat_logits = logits.reshape(-1, logits.size(-1))
            target_chunk = flat_targets[:, start:end].reshape(-1)
            weight_chunk = flat_weights[:, start:end].reshape(-1)
            loss_per_token = F.cross_entropy(
                flat_logits, target_chunk, reduction="none"
            )
            total_loss = total_loss + (loss_per_token * weight_chunk).sum()

            with torch.no_grad():
                pred_ids = torch.argmax(flat_logits, dim=-1)
                binary_eval_mask = weight_chunk > 0.0
                total_correct = total_correct + (
                    (pred_ids == target_chunk) & binary_eval_mask
                ).sum()

        loss = total_loss / valid_token_count
        accuracy = total_correct.float() / actual_token_count
        return loss, accuracy

    def compute_loss_and_accuracy(
        self,
        logits: torch.Tensor,
        training_view: DFlashTrainingView,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = training_view.target_ids.view(-1)
        flat_weights = training_view.weight_mask.view(-1)
        binary_eval_mask = flat_weights > 0.0

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        valid_token_count = flat_weights.sum() + 1e-6
        loss = (loss_per_token * flat_weights).sum() / valid_token_count

        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & binary_eval_mask
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        return loss, accuracy

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel block-wise training forward pass."""
        training_view = self.prepare_training_view(
            input_ids=input_ids,
            loss_mask=loss_mask,
        )
        logits = self.forward_from_view(
            hidden_states=hidden_states,
            training_view=training_view,
        )
        return self.compute_loss_and_accuracy(logits, training_view)
