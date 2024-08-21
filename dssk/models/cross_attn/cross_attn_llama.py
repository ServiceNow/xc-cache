# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# TODO: BE COMPLIANT WITH Apache License, Version 2.0 .
""" PyTorch LLaMA model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import transformers
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaAttention,
    LlamaRMSNorm,
    rotate_half,
    repeat_kv,
)
from accelerate import init_empty_weights


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


def apply_rotary_pos_emb(q_or_k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to either query or key tensors.

    Args:
        q_or_k (`torch.Tensor`): The query or k tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_or_k_embed = (q_or_k * cos) + (rotate_half(q_or_k) * sin)
    return q_or_k_embed


class MLP(nn.Module):
    def __init__(self, input_dim, intermediate_size, output_dim, config):
        super().__init__()
        self.config = config
        self.hidden_size = input_dim
        self.intermediate_size = intermediate_size
        self.output_size = output_dim
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.output_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaCrossAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__(config)
        self.config = config

        self.attention_dropout = config.cross_attn_dropout_prob
        self.base_hidden_size = config.hidden_size
        self.hidden_size = config.cross_attn_hidden_size
        self.num_heads = config.cross_attn_num_attention_heads
        self.cross_attn_shared_projections = config.cross_attn_shared_projections
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.cross_attn_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        # NOTE: self.is_causal is unused for now but will be useful if we decide to add flash_attn later.
        self.is_causal = False

        self.cross_attn_dropout = nn.Dropout(config.cross_attn_dropout_prob)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.base_hidden_size,
            self.num_heads * self.head_dim,
            bias=config.cross_attn_attention_bias,
        )
        self.k_proj = nn.Linear(
            self.base_hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.cross_attn_attention_bias,
        )
        self.v_proj = nn.Linear(
            self.base_hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.cross_attn_attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.base_hidden_size,
            bias=config.cross_attn_attention_bias,
        )
        self._init_rope()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cross_attn_position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        attn_implementation: str = "sdpa",
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()
        _, kv_seq_len, _ = encoder_hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(encoder_hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(encoder_hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, kv_seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, kv_seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        q_cos, q_sin = self.rotary_emb(query_states, position_ids)
        kv_cos, kv_sin = self.rotary_emb(value_states, cross_attn_position_ids)

        query_states = apply_rotary_pos_emb(query_states, q_cos, q_sin)
        key_states = apply_rotary_pos_emb(key_states, kv_cos, kv_sin)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        batch_size = hidden_states.size(0)
        batch_length = hidden_states.size(1)
        ctx_batch_len = encoder_hidden_states.size(1)

        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                (1, 1), device=hidden_states.device, dtype=hidden_states.dtype
            ).expand(batch_size, ctx_batch_len)

        if query_states.device.type == "cuda":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        if attn_implementation == "sdpa":

            # make 4d mask. From shape (bsz, kv_seq_len) to (bsz, 1, q_len, kv_seq_len)
            encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                encoder_attention_mask, dtype=hidden_states.dtype, tgt_len=batch_length
            )

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=encoder_attention_mask,
                dropout_p=self.config.cross_attn_dropout_prob if self.training else 0.0,
            )

            attn_weights = None
            if not output_attentions:
                Warning("output_attentions only has effect for eager attention implementation.")

        elif attn_implementation == "eager":

            if self.training:
                # Since we apply dropout on masks, we don't want its eval mode effect
                # Of using the expected value of its inputs, i.e., multiplying inputs by the
                # dropout probability.
                # We then only use this layer in training mode.
                original_dtype = encoder_attention_mask.dtype
                encoder_attention_mask = self.cross_attn_dropout(
                    encoder_attention_mask.float()
                ).to(original_dtype)

            # make 4d mask. From shape (bsz, kv_seq_len) to (bsz, 1, q_len, kv_seq_len)
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, dtype=hidden_states.dtype, tgt_len=batch_length
            )

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if encoder_attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {encoder_attention_mask.size()}"
                )
            attn_weights = attn_weights + encoder_attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.attention_dropout, training=self.training
            )
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            if not output_attentions:
                attn_weights = None

        else:
            raise NotImplementedError(
                "Unknown attention implementation. Set attn_implementation to either 'eager' or 'sdpa'."
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class CrossAttnLlamaBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()

        hidden_size = config.cross_attn_hidden_size
        base_hidden_size = config.hidden_size
        self.inner_dim = 4 * hidden_size
        self.initializer_range = config.initializer_range

        self.ln_1 = nn.LayerNorm(base_hidden_size, eps=config.rms_norm_eps)
        self.crossattention = LlamaCrossAttention(config)
        self.ln_2 = nn.LayerNorm(base_hidden_size, eps=config.rms_norm_eps)

        self.mlp = MLP(base_hidden_size, self.inner_dim, base_hidden_size, config)

        self.apply(self._init_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        # TODO: support loading of keys and values already computed in previous generations
        layer_past: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cross_attn_position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        attn_implementation: str = "sdpa",
    ) -> Union[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        cross_attn_outputs = self.crossattention(
            hidden_states,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            position_ids=position_ids,
            cross_attn_position_ids=cross_attn_position_ids,
            attn_implementation=attn_implementation,
        )

        cross_attn_outputs = cross_attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = cross_attn_outputs[1:]
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # residual connection
        hidden_states = cross_attn_outputs + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class CrossAttnLlama(LlamaForCausalLM):
    def __init__(
        self,
        model_id: str,
        n_cross_attn_layers: int = 2,
        cross_attn_layers_stride: int = 2,
        cross_attn_shared_weights: bool = True,
        cross_attn_dropout_prob: Optional[float] = 0.0,
        cross_attn_final_layer: bool = False,
        cross_attn_shared_projections: bool = False,
        cross_attn_hidden_size: Optional[int] = None,
        cross_attn_num_attention_heads: Optional[int] = None,
        cross_attn_num_key_value_heads: Optional[int] = None,
        randomly_initialize_decoder: bool = False,
        cross_attn_attention_bias: bool = False,
        cross_attn_skip_connections: bool = False,
        cache_dir: Optional[str] = None,
        max_len: Optional[int] = None,
        include_questions_on_contexts: Optional[bool] = None,
        chunked_contexts: Optional[bool] = None,
    ) -> None:
        config = transformers.AutoConfig.from_pretrained(model_id)
        print(config)
        with init_empty_weights():
            # We initialize an empty model to not waste ram.
            super().__init__(config)

        if cross_attn_hidden_size is None:
            cross_attn_hidden_size = config.hidden_size
        if cross_attn_num_attention_heads is None:
            cross_attn_num_attention_heads = config.num_attention_heads
        if cross_attn_num_key_value_heads is None:
            cross_attn_num_key_value_heads = cross_attn_num_attention_heads

        # We can optionally change the maximum decoder's input length.
        max_len = max_len if max_len is not None else config.max_position_embeddings

        config.update(
            {
                "n_cross_attn_layers": n_cross_attn_layers,
                "cross_attn_layers_stride": cross_attn_layers_stride,
                "cross_attn_shared_weights": cross_attn_shared_weights,
                "cross_attn_dropout_prob": cross_attn_dropout_prob,
                "cross_attn_final_layer": cross_attn_final_layer,
                "cross_attn_hidden_size": cross_attn_hidden_size,
                "cross_attn_num_attention_heads": cross_attn_num_attention_heads,
                "cross_attn_num_key_value_heads": cross_attn_num_key_value_heads,
                "cross_attn_shared_projections": cross_attn_shared_projections,
                "cross_attn_attention_bias": cross_attn_attention_bias,
                "cross_attn_skip_connections": cross_attn_skip_connections,
                "input_format_fn": "cross_uaf_question_in_context",
                "max_position_embeddings": max_len,
                "max_len": max_len,
                "include_questions_on_contexts": include_questions_on_contexts,
                "chunked_contexts": chunked_contexts,
            }
        )

        self.base_model_id = model_id
        self.n_decoder_layers = config.num_hidden_layers

        self.transformer, self.lm_head = self._make_base_decoder(cache_dir=cache_dir)

        self.cross_attn_layers = self._make_cross_attn_layers(config)

        self.cross_attn_final_layer = cross_attn_final_layer

        self.cross_attn_skip_connections = cross_attn_skip_connections

        if randomly_initialize_decoder:
            # Random init of all parameters.
            self.init_weights()

        delattr(self, "model")

    def _make_base_decoder(
        self,
        cache_dir=None,
    ) -> tuple[torch.nn.ModuleList, torch.nn.ModuleList]:
        base_decoder = LlamaForCausalLM.from_pretrained(self.base_model_id, cache_dir=cache_dir)

        return base_decoder.model, base_decoder.lm_head

    def _make_cross_attn_layers(
        self,
        decoder_config: LlamaConfig,
    ) -> torch.nn.ModuleList:
        """Introduces cross-attn layers."""

        n_layers = decoder_config.n_cross_attn_layers
        layer_stride = decoder_config.cross_attn_layers_stride
        shared_weights = decoder_config.cross_attn_shared_weights

        assert n_layers > 0
        assert n_layers <= self.n_decoder_layers
        assert self.n_decoder_layers - layer_stride * n_layers >= 0

        cross_attn_layer_list = [torch.nn.Identity() for _ in range(self.n_decoder_layers)]
        if shared_weights:
            base_cross_attn_layer = CrossAttnLlamaBlock(
                decoder_config, layer_idx=self.n_decoder_layers - 1
            )

        for layer_idx in range(
            self.n_decoder_layers - 1,
            self.n_decoder_layers - layer_stride * n_layers - 1,
            -layer_stride,
        ):
            if shared_weights:
                cross_attn_layer_list[layer_idx] = base_cross_attn_layer
            else:
                cross_attn_layer_list[layer_idx] = CrossAttnLlamaBlock(
                    decoder_config, layer_idx=layer_idx
                )

        return torch.nn.ModuleList(cross_attn_layer_list)

    def prepare_for_training(
        self, train_all_params: bool = False, use_gradient_checkpointing: bool = False
    ) -> "CrossAttnLlama":
        """
        Prepare model for training by setting which params require gradients.
        Only cross-attn layers will require gradients by default, and all params will
        if train_all_params is set.
        """

        self._train_all_params = train_all_params

        if not isinstance(train_all_params, bool):
            raise ValueError("train_all_params is expected to be boolean")
        # We set both requires_grad and requires_grad_ to ensure the desired
        # behaviour will be propagated no matter which attribute is used downstream.
        for _, v in self.named_parameters():
            v.requires_grad_(train_all_params)
        for _, v in self.cross_attn_layers.named_parameters():
            v.requires_grad_(True)

        if use_gradient_checkpointing:
            # For backward compatibility
            if hasattr(self, "enable_input_require_grads"):
                self.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            self.gradient_checkpointing_enable()
            self.gradient_checkpointing = True
        else:
            self.gradient_checkpointing = False

        return self

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        context_input_ids=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # retrieve encoded context
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        encoder_attention_mask = kwargs.get("encoder_attention_mask", None)

        if encoder_hidden_states is None:
            if context_input_ids is None:
                raise ValueError(
                    "Either 'context_ids' with tokenized context or 'encoder_hidden_states' with encoded context must be passed to generate()."
                )

            if encoder_attention_mask is None:
                Warning(
                    "Missing 'encoder_attention_mask' argument: no padded attention mask is provided for the context. Setting it to default full mask."
                )
                encoder_attention_mask = torch.ones_like(context_input_ids)

            with torch.no_grad():
                encoder_hidden_states, encoder_attention_mask = self.encode(
                    input_ids=context_input_ids,
                    attention_mask=encoder_attention_mask,
                )

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": encoder_attention_mask,
            }
        )

        return model_inputs

    def train(self, mode: bool = True) -> "CrossAttnLlama":
        """Sets the module in training mode.

        Overrides train() to set only cross-attn params in train mode.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if hasattr(self, "_train_all_params") and self._train_all_params:
            for module in self.children():
                module.train(mode)
        else:
            for module in self.children():
                module.train(False)
            if mode:
                for module in self.cross_attn_layers.children():
                    module.train(mode)
        return self

    def forward(  # noqa: C901 'CrossAttnLlama.forward' is too complex (27)
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        context_input_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        encoder_hidden_states: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        encoder_attention_mask: Optional[Union[torch.LongTensor, List[torch.LongTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cross_attn_implementation: str = "sdpa",
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if encoder_hidden_states is None:
            if context_input_ids is None:
                raise ValueError(
                    "Either 'context_ids' with tokenized context or 'encoder_hidden_states' with encoded context must be passed to generate()."
                )

            encoder_hidden_states, encoder_attention_mask = self.encode(
                input_ids=context_input_ids,
                attention_mask=encoder_attention_mask,
            )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            Warning(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.transformer.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, Cache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        try:
            context_length = encoder_hidden_states.size(1)
        except AttributeError:
            context_length = 0
            for context_chunk in encoder_hidden_states:
                context_length += context_chunk.size(1)

        cross_attn_position_ids = torch.arange(
            context_length, device=inputs_embeds.device
        ).unsqueeze(0)

        causal_mask = self.transformer._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # cross_attn_layer_idx tracks the idx of the current cross-attn layer as we forward pass.
        # It's only going to be used if encoder_hidden_states contains a list of different tensors
        # to be cross-attended to.
        cross_attn_layer_idx = 0

        for idx, decoder_layer in enumerate(self.transformer.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Cross-attend first if not self.cross_attn_final_layer.
            if (
                not isinstance(self.cross_attn_layers[idx], torch.nn.Identity)
                and not self.cross_attn_final_layer
            ):
                cross_attn_outputs = self.cross_attn_forward(
                    hidden_states=hidden_states,
                    layer_idx=idx,
                    cross_attn_layer_idx=cross_attn_layer_idx,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    position_ids=position_ids,
                    cross_attn_position_ids=cross_attn_position_ids,
                    output_attentions=output_attentions,
                    cross_attn_implementation=cross_attn_implementation,
                )
                cross_attn_layer_idx += 1

                hidden_states = cross_attn_outputs[0]

            if self.gradient_checkpointing and self.training:
                layer_outputs = self.transformer._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # Cross-attend after self-attn if self.cross_attn_final_layer.
            if (
                not isinstance(self.cross_attn_layers[idx], torch.nn.Identity)
                and self.cross_attn_final_layer
            ):
                cross_attn_outputs = self.cross_attn_forward(
                    hidden_states=hidden_states,
                    layer_idx=idx,
                    cross_attn_layer_idx=cross_attn_layer_idx,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    position_ids=position_ids,
                    cross_attn_position_ids=cross_attn_position_ids,
                    output_attentions=output_attentions,
                    cross_attn_implementation=cross_attn_implementation,
                )
                cross_attn_layer_idx += 1

                hidden_states = cross_attn_outputs[0]

        hidden_states = self.transformer.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache)
                else next_decoder_cache
            )

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = tuple(
                v
                for v in [logits, hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    def cross_attn_forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_idx: int,
        cross_attn_layer_idx: Optional[
            int
        ] = -1,  # This is only used if encoder_hidden_states is a list of tensors to cross-attend to.
        encoder_hidden_states: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        encoder_attention_mask: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cross_attn_position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        cross_attn_implementation: str = "sdpa",
    ) -> Union[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if isinstance(encoder_hidden_states, list):
            # if encoder_hidden_states is a list, we expect its length to match the number of cross-attn layers
            # and cross_attn_layer_idx will be used to indicate which element should be used at the current layer.
            encoder_hidden_states = encoder_hidden_states[cross_attn_layer_idx]
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask[cross_attn_layer_idx]

        if self.gradient_checkpointing and self.training:
            # TODO(Joao): Add this to the forward hook defined in self.prepare_for_training.
            hidden_states.requires_grad_(True)
            encoder_hidden_states.requires_grad_(True)

            cross_attn_outputs = torch.utils.checkpoint.checkpoint(
                self.cross_attn_layers[layer_idx].__call__,
                hidden_states,
                None,
                None,
                encoder_hidden_states,
                encoder_attention_mask,
                position_ids,
                cross_attn_position_ids,
                None,
                output_attentions,
                cross_attn_implementation,
            )
        else:
            cross_attn_outputs = self.cross_attn_layers[layer_idx](
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                position_ids=position_ids,
                cross_attn_position_ids=cross_attn_position_ids,
                use_cache=None,
                output_attentions=output_attentions,
                attn_implementation=cross_attn_implementation,
            )

        if self.cross_attn_skip_connections:
            return [hidden_states + cross_attn_outputs[0]]

        return cross_attn_outputs

    def encode(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if isinstance(input_ids, list):
            # If we get a list of contexts, we embed each context
            # indepedently and concatenate afterward.

            # Input_ids and att_mask are expected to be such that:
            # [[B, T, D]]*num_chunks

            num_chunks = len(input_ids)

            input_ids = torch.cat(input_ids, dim=0)
            cat_attention_mask = torch.cat(attention_mask, dim=0)

            encoder_hidden_states = self.transformer(
                input_ids=input_ids,
                attention_mask=cat_attention_mask,
            ).last_hidden_state.detach()

            encoder_hidden_states = torch.cat(
                torch.chunk(encoder_hidden_states, num_chunks, dim=0), dim=1
            )
            attention_mask = torch.cat(attention_mask, dim=1)

        else:
            encoder_hidden_states = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state.detach()

        return encoder_hidden_states, attention_mask
