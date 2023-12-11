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
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention, 
    LlamaRMSNorm, 
    LlamaForCausalLM, 
    rotate_half, 
    repeat_kv, 
)
from accelerate import init_empty_weights


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)

def apply_rotary_pos_emb(q_or_k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to either query or key tensors.

    Args:
        q_or_k (`torch.Tensor`): The query or k tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
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
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
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

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.base_hidden_size, self.num_heads * self.head_dim, bias=config.cross_attn_attention_bias)
        self.k_proj = nn.Linear(self.base_hidden_size, self.num_key_value_heads * self.head_dim, bias=config.cross_attn_attention_bias)
        self.v_proj = nn.Linear(self.base_hidden_size, self.num_key_value_heads * self.head_dim, bias=config.cross_attn_attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.base_hidden_size, bias=config.cross_attn_attention_bias)
        self._init_rope()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()
        _, kv_seq_len, _ = encoder_hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(encoder_hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(encoder_hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        q_cos, q_sin = self.rotary_emb(query_states, seq_len=q_len)
        kv_cos, kv_sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states= apply_rotary_pos_emb(query_states, q_cos, q_sin, position_ids)
        key_states = apply_rotary_pos_emb(key_states, kv_cos, kv_sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        batch_size = hidden_states.size(0)
        batch_length = hidden_states.size(1)
        ctx_batch_len = encoder_hidden_states.size(1)

        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                (batch_size, 1, batch_length, ctx_batch_len), device=hidden_states.device
            )

        attention_mask = encoder_attention_mask.to(
            dtype=torch.bool, device=hidden_states.device
        )

        # make 4d mask. From shape (bsz, kv_seq_len) to (bsz, 1, q_len, kv_seq_len)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat([1, 1, q_len, 1])

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

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
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
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
        cross_attn_final_layer: Optional[bool] = False,
        cross_attn_shared_projections: Optional[bool] = False,
        cross_attn_hidden_size: Optional[int] = None,
        cross_attn_num_attention_heads: Optional[int] = None,
        cross_attn_num_key_value_heads: Optional[int] = None,
        randomly_initialize_decoder: Optional[bool] = False,
        cross_attn_attention_bias: Optional[bool] = False,
    ) -> None:
        config = transformers.AutoConfig.from_pretrained(model_id)
        with init_empty_weights():
            # We initialize an empty model to not waste ram.
            super().__init__(config)

        if cross_attn_hidden_size is None:
            cross_attn_hidden_size = config.hidden_size
        if cross_attn_num_attention_heads is None:
            cross_attn_num_attention_heads = config.num_attention_heads
        if cross_attn_num_key_value_heads is None:
            cross_attn_num_key_value_heads = cross_attn_num_attention_heads

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
            }
        )

        self.base_model_id = model_id
        self.n_decoder_layers = config.num_hidden_layers

        self.transformer, self.lm_head = self._make_base_decoder()

        self.cross_attn_layers = self._make_cross_attn_layers(config)

        self.cross_attn_final_layer = cross_attn_final_layer

        if randomly_initialize_decoder:
            # Random init of all parameters.
            self.init_weights()
        
        delattr(self, "model")
        
    def _make_base_decoder(
        self,
    ) -> tuple[torch.nn.ModuleList, torch.nn.ModuleList]:
        base_decoder = LlamaForCausalLM.from_pretrained(
            self.base_model_id,
        )
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
        self, input_ids, past_key_values=None, inputs_embeds=None, context_ids=None, **kwargs
    ):
        """
        Prepare inputs for inference, which might require encoding the context.
        """

        token_type_ids = kwargs.get("token_type_ids", None)

        # TODO: support passing of already computed input projections in previous generations
        # only last token for inputs_ids is passed if defined in kwargs
        # if past_key_values:
        #     input_ids = input_ids[:, -1].unsqueeze(-1)
        #     if token_type_ids is not None:
        #         token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        past_key_values = None

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # retrieve encoded context
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        encoder_attention_mask = kwargs.get("encoder_attention_mask", None)

        if encoder_hidden_states is None:
            if context_ids is None:
                raise ValueError(
                    "Either 'context_ids' with tokenized context or 'encoder_hidden_states' with encoded context must be passed to generate()."
                )

            if encoder_attention_mask is None:
                Warning(
                    "Missing 'encoder_attention_mask' argument: no padded attention mask is provided for the context. Setting it to default full mask."
                )
                encoder_attention_mask = torch.ones_like(context_ids)

            with torch.no_grad():
                encoder_hidden_states = self.transformer(
                    input_ids=context_ids,
                    attention_mask=encoder_attention_mask,
                ).last_hidden_state.detach()

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": encoder_attention_mask,
            }
        )
        return model_inputs

    def train(self, mode: Optional[bool] = True) -> "CrossAttnLlama":
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.transformer.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
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
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.transformer.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # Cross-attend first if not self.cross_attn_final_layer.
            if (
                not isinstance(self.cross_attn_layers[idx], torch.nn.Identity)
                and not self.cross_attn_final_layer
            ):
                cross_attn_outputs = self.cross_attn_forward(
                    hidden_states=hidden_states,
                    layer_idx=idx,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

                hidden_states = cross_attn_outputs[0]

            if self.gradient_checkpointing and self.training:
                layer_outputs = self.transformer._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

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
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

                hidden_states = cross_attn_outputs[0]

        hidden_states = self.transformer.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
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
            output = tuple(v for v in [logits, hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

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

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> Union[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:

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
                use_cache,
                output_attentions
            )
        else:
            cross_attn_outputs = self.cross_attn_layers[layer_idx](
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        return cross_attn_outputs
