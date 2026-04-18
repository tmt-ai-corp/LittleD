import heapq
import time
from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def find_first_stop_sequence(
    token_ids: torch.Tensor,
    stop_token_sequences: Optional[list[list[int]]],
) -> Optional[int]:
    if not stop_token_sequences:
        return None

    token_list = token_ids.tolist()
    best_index = None
    for stop_sequence in stop_token_sequences:
        if not stop_sequence:
            continue
        stop_len = len(stop_sequence)
        for idx in range(0, len(token_list) - stop_len + 1):
            if token_list[idx : idx + stop_len] == stop_sequence:
                best_index = idx if best_index is None else min(best_index, idx)
                break
    return best_index


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor]:
    """Build a DDTree candidate tree from per-position DFlash logits.

    The returned tensors live on CPU intentionally: the tree is tiny relative to
    model activations and the heap search is control-flow heavy.
    """
    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
        )

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids = top_token_ids.to(device="cpu", dtype=torch.long)

    first_logw = float(top_log_probs[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]

    node_token_ids = torch.empty(budget, dtype=torch.long)
    node_depths = torch.empty(budget, dtype=torch.long)
    parents = [-1]
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids[node_count] = token_id
        node_depths[node_count] = depth
        parents.append(parent_index)
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw
                - float(top_log_probs[depth - 1, rank])
                + float(top_log_probs[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap,
                (
                    -sibling_logw,
                    sibling_ranks,
                    parent_index,
                    depth,
                    rank + 1,
                    sibling_logw,
                ),
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs[depth, 0])
            heapq.heappush(
                heap,
                (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw),
            )

    current_length = 1 + node_count
    visibility = torch.zeros((current_length, current_length), dtype=torch.bool)
    visibility[0, 0] = True
    for index in range(1, current_length):
        parent_index = parents[index]
        visibility[index, :index] = visibility[parent_index, :index]
        visibility[index, index] = True

    return (
        node_token_ids[:node_count],
        node_depths[:node_count],
        parents,
        child_maps,
        visibility,
    )


def compile_ddtree_tree(
    *,
    root_token_id: torch.Tensor,
    start: int,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility_cpu: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    verify_input_ids_buffer: torch.Tensor,
    verify_position_ids_buffer: torch.Tensor,
    attention_mask_buffer: torch.Tensor,
    tree_visibility_buffer: torch.Tensor,
    previous_tree_start: int,
    previous_tree_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    current_length = 1 + int(node_token_ids.numel())

    if previous_tree_length > 0:
        attention_mask_buffer[
            0,
            0,
            :previous_tree_length,
            previous_tree_start : previous_tree_start + previous_tree_length,
        ] = 0

    verify_input_ids = verify_input_ids_buffer[:, :current_length]
    verify_input_ids[0, 0] = root_token_id
    if current_length > 1:
        verify_input_ids[0, 1:current_length].copy_(node_token_ids)

    verify_position_ids = verify_position_ids_buffer[:, :current_length]
    verify_position_ids[0, 0] = start
    if current_length > 1:
        verify_position_ids[0, 1:current_length].copy_(node_depths)
        verify_position_ids[0, 1:current_length].add_(start)

    visibility = tree_visibility_buffer[:current_length, :current_length]
    visibility.copy_(visibility_cpu)

    tree_block = attention_mask_buffer[
        0,
        0,
        :current_length,
        past_length : past_length + current_length,
    ]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(visibility, 0)

    attention_mask = attention_mask_buffer[
        :,
        :,
        :current_length,
        : past_length + current_length,
    ]
    return verify_input_ids, verify_position_ids, attention_mask, past_length, current_length


def follow_verified_tree(
    child_maps: list[dict[int, int]],
    posterior: torch.Tensor,
) -> tuple[list[int], int]:
    posterior_tokens = posterior[0].tolist()
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token


def _compact_appended_window(
    cache_tensor: torch.Tensor,
    past_length: int,
    keep_current_indices: torch.Tensor,
) -> None:
    current_length = cache_tensor.shape[-2] - past_length
    if current_length <= 0:
        return

    keep_count = keep_current_indices.numel()
    if keep_count == 0 or keep_count == current_length:
        return

    kept_tail = cache_tensor.narrow(-2, past_length, current_length).index_select(
        -2, keep_current_indices
    )
    cache_tensor.narrow(-2, past_length, keep_count).copy_(kept_tail)


def compact_dynamic_cache(
    past_key_values: DynamicCache,
    past_length: int,
    keep_current_indices: list[int],
) -> None:
    if len(keep_current_indices) == 0:
        past_key_values.crop(past_length)
        return

    keep_tensor_by_device: dict[torch.device, torch.Tensor] = {}

    def get_keep_tensor(device: torch.device) -> torch.Tensor:
        if device not in keep_tensor_by_device:
            keep_tensor_by_device[device] = torch.tensor(
                keep_current_indices,
                dtype=torch.long,
                device=device,
            )
        return keep_tensor_by_device[device]

    if hasattr(past_key_values, "key_cache") and hasattr(
        past_key_values, "value_cache"
    ):
        for key_cache, value_cache in zip(
            past_key_values.key_cache,
            past_key_values.value_cache,
        ):
            keep_tensor = get_keep_tensor(key_cache.device)
            _compact_appended_window(key_cache, past_length, keep_tensor)
            _compact_appended_window(value_cache, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if not hasattr(layer, "keys") or layer.keys is None:
                continue
            if layer.keys.numel() == 0:
                continue
            keep_tensor = get_keep_tensor(layer.keys.device)
            _compact_appended_window(layer.keys, past_length, keep_tensor)
            _compact_appended_window(layer.values, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    raise RuntimeError("Unsupported DynamicCache layout for DDTree cache compaction.")


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = [] if output_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return hidden_states, tuple(all_hidden_states)
        return hidden_states

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: Optional[list[int]],
        temperature: float,
        return_stats: bool = False,
        stop_token_sequences: Optional[list[list[int]]] = None,
        apply_ddtree: bool = False,
        ddtree_size: int = 32,
    ):
        if apply_ddtree:
            return self.spec_generate_ddtree(
                target=target,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=temperature,
                return_stats=return_stats,
                stop_token_sequences=stop_token_sequences,
                tree_budget=ddtree_size,
            )

        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill stage
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        # Decode stage
        acceptance_lengths = []
        stop_sequence_start = None
        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, past_key_values_draft.get_seq_length() : start + block_size
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -block_size + 1 :, :]
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, : acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length + 1)

            search_end = min(start + 1, max_length, output_ids.shape[1])
            stop_sequence_index = find_first_stop_sequence(
                output_ids[0, num_input_tokens:search_end],
                stop_token_sequences,
            )
            if stop_sequence_index is not None:
                stop_sequence_start = num_input_tokens + stop_sequence_index
                break

            if stop_token_ids is not None:
                stop_ids = torch.tensor(stop_token_ids, device=output_ids.device)
                if torch.isin(
                    output_ids[0, num_input_tokens:search_end],
                    stop_ids,
                ).any():
                    break
        output_ids = output_ids[:, :max_length]
        if stop_sequence_start is not None:
            output_ids = output_ids[:, :stop_sequence_start]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_sequence_start is None and stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        if not return_stats:
            return output_ids

        num_new_tokens = max(output_ids.shape[1] - num_input_tokens, 0)
        accept_length = (
            sum(acceptance_lengths) / len(acceptance_lengths)
            if acceptance_lengths
            else 1.0
        )
        return output_ids, {
            "acceptance_lengths": acceptance_lengths,
            "num_new_tokens": num_new_tokens,
            "num_speculation_steps": len(acceptance_lengths),
            "accept_length": accept_length,
        }

    @torch.inference_mode()
    def spec_generate_ddtree(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: Optional[list[int]],
        temperature: float,
        return_stats: bool = False,
        stop_token_sequences: Optional[list[list[int]]] = None,
        tree_budget: int = 32,
    ):
        self.eval()
        block_size = self.block_size
        if block_size <= 1:
            return self.spec_generate(
                target=target,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=temperature,
                return_stats=return_stats,
                stop_token_sequences=stop_token_sequences,
                apply_ddtree=False,
            )

        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        draft_horizon = block_size - 1
        tree_budget = max(int(tree_budget), 0)
        max_tree_nodes = 1 + tree_budget
        target_dtype = getattr(target, "dtype", next(target.parameters()).dtype)

        output_ids = torch.full(
            (1, max_length + max_tree_nodes),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)

        verify_input_ids_buffer = torch.empty(
            (1, max_tree_nodes),
            dtype=torch.long,
            device=target.device,
        )
        verify_position_ids_buffer = torch.empty(
            (1, max_tree_nodes),
            dtype=torch.long,
            device=target.device,
        )
        attention_mask_buffer = torch.zeros(
            (1, 1, max_tree_nodes, max_length + max_tree_nodes),
            dtype=target_dtype,
            device=target.device,
        )
        tree_visibility_buffer = torch.empty(
            (max_tree_nodes, max_tree_nodes),
            dtype=torch.bool,
            device=target.device,
        )

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        acceptance_lengths = []
        stop_sequence_start = None
        previous_tree_start = 0
        previous_tree_length = 0
        ddtree_stage_times = {
            "draft": 0.0,
            "tree_build": 0.0,
            "tree_compile": 0.0,
            "verify": 0.0,
            "commit": 0.0,
        }
        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            root_token = block_output_ids[:, :1]

            stage_start = time.perf_counter()
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, past_key_values_draft.get_seq_length() : start + block_size
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -draft_horizon:, :]
            )
            past_key_values_draft.crop(start)
            ddtree_stage_times["draft"] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            (
                node_token_ids,
                node_depths,
                _parents,
                child_maps,
                visibility_cpu,
            ) = build_ddtree_tree(draft_logits[0], tree_budget)
            ddtree_stage_times["tree_build"] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            (
                verify_input_ids,
                verify_position_ids,
                verify_attention_mask,
                previous_tree_start,
                previous_tree_length,
            ) = compile_ddtree_tree(
                root_token_id=root_token[0, 0],
                start=start,
                node_token_ids=node_token_ids,
                node_depths=node_depths,
                visibility_cpu=visibility_cpu,
                past_length=start,
                dtype=target_dtype,
                verify_input_ids_buffer=verify_input_ids_buffer,
                verify_position_ids_buffer=verify_position_ids_buffer,
                attention_mask_buffer=attention_mask_buffer,
                tree_visibility_buffer=tree_visibility_buffer,
                previous_tree_start=previous_tree_start,
                previous_tree_length=previous_tree_length,
            )
            ddtree_stage_times["tree_compile"] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            output = target(
                verify_input_ids,
                position_ids=verify_position_ids,
                attention_mask=verify_attention_mask,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            ddtree_stage_times["verify"] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            posterior = sample(output.logits, temperature)
            accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
            accepted_index_tensor = torch.tensor(
                accepted_indices,
                dtype=torch.long,
                device=verify_input_ids.device,
            )
            accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

            output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
            output_ids[:, start + len(accepted_indices)] = next_token

            compact_dynamic_cache(past_key_values_target, start, accepted_indices)
            target_hidden = extract_context_feature(
                output.hidden_states,
                self.target_layer_ids,
            ).index_select(1, accepted_index_tensor)

            acceptance_lengths.append(len(accepted_indices))
            start += len(accepted_indices)
            ddtree_stage_times["commit"] += time.perf_counter() - stage_start

            search_end = min(start + 1, max_length, output_ids.shape[1])
            stop_sequence_index = find_first_stop_sequence(
                output_ids[0, num_input_tokens:search_end],
                stop_token_sequences,
            )
            if stop_sequence_index is not None:
                stop_sequence_start = num_input_tokens + stop_sequence_index
                break

            if stop_token_ids is not None:
                stop_ids = torch.tensor(stop_token_ids, device=output_ids.device)
                if torch.isin(
                    output_ids[0, num_input_tokens:search_end],
                    stop_ids,
                ).any():
                    break

        output_ids = output_ids[:, :max_length]
        if stop_sequence_start is not None:
            output_ids = output_ids[:, :stop_sequence_start]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_sequence_start is None and stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:],
                stop_token_ids,
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        if not return_stats:
            return output_ids

        num_new_tokens = max(output_ids.shape[1] - num_input_tokens, 0)
        accept_length = (
            sum(acceptance_lengths) / len(acceptance_lengths)
            if acceptance_lengths
            else 1.0
        )
        return output_ids, {
            "acceptance_lengths": acceptance_lengths,
            "num_new_tokens": num_new_tokens,
            "num_speculation_steps": len(acceptance_lengths),
            "accept_length": accept_length,
            "ddtree_size": tree_budget,
            "ddtree_stage_times": ddtree_stage_times,
        }
