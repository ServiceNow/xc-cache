# This module is mostly adapted from transformers.models.gpt_bigcode.modeling_gpt_bigcode
# We subclass and modify GPTBigCodeAttention and GPTBigCodeForCausalLM so that we can include cross-attn layers
# And add functionality for the cases where we want to train only those layers.

import torch
import torch.utils.checkpoint
from torch import nn
import transformers
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import (
    GPTBigCodeForCausalLM,
    GPTBigCodeAttention,
    upcast_softmax,
    upcast_masked_softmax,
)
from transformers.models.gpt_bigcode.configuration_gpt_bigcode import GPTBigCodeConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import List, Optional, Tuple, Union
from accelerate import init_empty_weights


class MLP(nn.Module):
    def __init__(self, input_dim, intermediate_size, output_dim, config):
        super().__init__()
        self.c_fc = nn.Linear(input_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, output_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2MLP.forward
    def forward(self, hidden_states: Optional[Tuple[torch.Tensor]]) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CrossAttention(GPTBigCodeAttention):
    def __init__(
        self,
        config: GPTBigCodeConfig,
        layer_idx: int,
    ) -> None:
        super(GPTBigCodeAttention, self).__init__()
        self.mask_value = None

        self.base_embed_dim = config.hidden_size
        self.embed_dim = config.cross_attn_hidden_size
        self.num_heads = config.cross_attn_num_attention_heads
        self.cross_attn_shared_projections = config.cross_attn_shared_projections
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_dim = self.num_heads * self.head_dim
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = (
            config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        )

        self.v_attn = nn.Linear(self.base_embed_dim, self.embed_dim)
        if self.cross_attn_shared_projections:
            self.q_k_attn = nn.Linear(self.base_embed_dim, self.embed_dim)
        else:
            self.q_attn = nn.Linear(self.base_embed_dim, self.embed_dim)
            self.k_attn = nn.Linear(self.base_embed_dim, self.embed_dim)

        self.c_proj = nn.Linear(self.embed_dim, self.base_embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.cross_attn_dropout = nn.Dropout(config.cross_attn_dropout_prob)

        self.layer_idx = layer_idx

    def _attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        dtype = query.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype
        upcast = dtype != softmax_dtype

        if self.scale_attn_weights:
            unscale = self.layer_idx + 1 if self.scale_attention_softmax_in_fp32 and upcast else 1
            scale_factor = unscale**-1
            scale_factor /= self.head_dim**0.5
        else:
            scale_factor = 1.0

        # MQA models: (batch_size, query_length, num_heads * head_dim)
        # MHA models: (batch_size, num_heads, query_length, head_dim)
        query_shape = query.shape
        batch_size = query_shape[0]
        key_length = key.size(-1)

        # (batch_size, num_heads, query_length, head_dim) x (batch_size, num_heads, head_dim, key_length)
        # -> (batch_size, num_heads, query_length, key_length)
        query_length = query_shape[2]
        attn_shape = (batch_size, self.num_heads, query_length, key_length)
        attn_view = (batch_size * self.num_heads, query_length, key_length)
        # Always copies
        query = query.reshape(batch_size * self.num_heads, query_length, self.head_dim)
        # No copy when layer_past is provided.
        key = key.reshape(batch_size * self.num_heads, self.head_dim, key_length)

        attn_weights = torch.empty(attn_view, device=query.device, dtype=query.dtype)
        if query.device.type == "cpu":
            # This is needed because of a bug in pytorch https://github.com/pytorch/pytorch/issues/80588.
            # The bug was fixed in https://github.com/pytorch/pytorch/pull/96086,
            # but the fix has not been released as of pytorch version 2.0.0.
            attn_weights.zero_()
            beta = 1
        else:
            beta = 0
        attn_weights = torch.baddbmm(attn_weights, query, key, beta=beta, alpha=scale_factor).view(
            attn_shape
        )

        if upcast:
            # Use a fused kernel to prevent a large overhead from casting and scaling.
            # Sub-optimal when the key length is not a multiple of 8.
            if attention_mask is None:
                attn_weights = upcast_softmax(attn_weights, unscale, softmax_dtype)
            else:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)
                attn_weights = upcast_masked_softmax(
                    attn_weights, attention_mask, mask_value, unscale, softmax_dtype
                )
        else:
            if attention_mask is not None:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)

                # The fused kernel is very slow when the key length is not a multiple of 8, so we skip fusion.
                attn_weights = torch.where(attention_mask, attn_weights, mask_value)

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    # Head splitting is copied from GPT2 to circumvent errors when using
    # GPTBigCodeAttention in MHA mode.
    def _split_heads(
        self, tensor: torch.Tensor, num_heads: int, attn_head_size: int
    ) -> torch.Tensor:
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
    ]:
        if self.cross_attn_shared_projections:
            query = self.q_k_attn(hidden_states)
            key = self.q_k_attn(encoder_hidden_states)
        else:
            query = self.q_attn(hidden_states)
            key = self.k_attn(encoder_hidden_states)
        value = self.v_attn(encoder_hidden_states)
        batch_size = hidden_states.size(0)
        batch_length = hidden_states.size(1)
        hidden_states_shape = [batch_size, batch_length, query.size(-1)]

        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                (batch_size, batch_length), device=hidden_states.device
            )

        if self.training:
            # Since we apply dropout on masks, we don't want its eval mode effect
            # Of using the expected value of its inputs, i.e., multiplying inputs by the
            # dropout probability.
            # We then only use this layer in training mode.
            encoder_attention_mask = self.cross_attn_dropout(encoder_attention_mask)

        attention_mask = encoder_attention_mask.view(batch_size, 1, -1).to(
            dtype=torch.bool, device=hidden_states.device
        )

        # MHA models: (batch_size, n_heads, query_length, key_length)
        attention_mask = attention_mask.unsqueeze(1)

        # TODO: support loading of keys, values and queries already computed in previous generations
        # if layer_past is not None:
        #     key_value = torch.cat((layer_past, key_value), dim=-2)
        # present = key_value if use_cache else None
        present = None

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output, attn_weights = self._attn(
            query, key.transpose(-1, -2), value, attention_mask, head_mask
        )

        attn_output = attn_output.transpose(1, 2).reshape(hidden_states_shape)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class CrossAttnGPTBigCodeBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()

        hidden_size = config.cross_attn_hidden_size
        base_hidden_size = config.hidden_size
        self.inner_dim = 4 * hidden_size
        self.initializer_range = config.initializer_range

        self.ln_1 = nn.LayerNorm(base_hidden_size, eps=config.layer_norm_epsilon)
        self.crossattention = CrossAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(base_hidden_size, eps=config.layer_norm_epsilon)

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


class CrossAttnGPTBigCode(GPTBigCodeForCausalLM):
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
        cross_attn_skip_connections: Optional[bool] = False,
        randomly_initialize_decoder: Optional[bool] = False,
        cache_dir: Optional[str] = None,
        max_len: int = -1,
    ) -> None:
        config = transformers.AutoConfig.from_pretrained(model_id)
        with init_empty_weights():
            # We initialize an empty model to not waste ram.
            super().__init__(config)

        if cross_attn_hidden_size is None:
            cross_attn_hidden_size = config.hidden_size
        if cross_attn_num_attention_heads is None:
            cross_attn_num_attention_heads = config.num_attention_heads

        config.update(
            {
                "n_cross_attn_layers": n_cross_attn_layers,
                "cross_attn_layers_stride": cross_attn_layers_stride,
                "cross_attn_shared_weights": cross_attn_shared_weights,
                "cross_attn_dropout_prob": cross_attn_dropout_prob,
                "cross_attn_final_layer": cross_attn_final_layer,
                "cross_attn_hidden_size": cross_attn_hidden_size,
                "cross_attn_num_attention_heads": cross_attn_num_attention_heads,
                "cross_attn_shared_projections": cross_attn_shared_projections,
                "cross_attn_skip_connections": cross_attn_skip_connections,
                "input_format_fn": "cross_uaf_question_in_context",
                "max_len": max_len,
            }
        )

        self.base_model_id = model_id
        self.n_decoder_layers = config.n_layer

        self.transformer, self.lm_head = self._make_base_decoder(cache_dir)

        self.cross_attn_layers = self._make_cross_attn_layers(config)

        self.cross_attn_final_layer = cross_attn_final_layer

        self.cross_attn_skip_connections = cross_attn_skip_connections

        if randomly_initialize_decoder:
            # Random init of all parameters.
            self.init_weights()

    def _make_base_decoder(
        self, cache_dir=None
    ) -> tuple[torch.nn.ModuleList, torch.nn.ModuleList]:
        base_decoder = GPTBigCodeForCausalLM.from_pretrained(
            self.base_model_id, cache_dir=cache_dir
        )
        return base_decoder.transformer, base_decoder.lm_head

    def _make_cross_attn_layers(
        self,
        decoder_config: GPTBigCodeConfig,
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
            base_cross_attn_layer = CrossAttnGPTBigCodeBlock(
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
                cross_attn_layer_list[layer_idx] = CrossAttnGPTBigCodeBlock(
                    decoder_config, layer_idx=layer_idx
                )

        return torch.nn.ModuleList(cross_attn_layer_list)

    def prepare_for_training(
        self, train_all_params: bool = False, use_gradient_checkpointing: bool = False
    ) -> "CrossAttnGPTBigCode":
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
            v.requires_grad = train_all_params
            v.requires_grad_ = train_all_params
        for _, v in self.cross_attn_layers.named_parameters():
            v.requires_grad = True
            v.requires_grad_ = True

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

    def train(self, mode: Optional[bool] = True) -> "CrossAttnGPTBigCode":
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

    def forward(  # noqa: C901 'CrossAttnGPTBigCode.forward' is too complex (27)
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.transformer.h))
        else:
            past_length = past_key_values[0].size(-2)

        if attention_mask is not None and len(attention_mask.shape) == 2 and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_length > 0:
                position_ids = position_ids[:, past_length : input_shape[-1] + past_length :]
        elif position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Self-attention mask.
        query_length = input_shape[-1]
        key_length = past_length + query_length
        self_attention_mask = self.transformer.bias[
            None, key_length - query_length : key_length, :key_length
        ]

        if attention_mask is not None:
            self_attention_mask = self_attention_mask * attention_mask.view(batch_size, 1, -1).to(
                dtype=torch.bool, device=self_attention_mask.device
            )

        # MQA models: (batch_size, query_length, n_heads, key_length)
        # MHA models: (batch_size, n_heads, query_length, key_length)
        attention_mask = self_attention_mask.unsqueeze(2 if self.transformer.multi_query else 1)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.transformer.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.transformer.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = [] if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None

        # cross_attn_layer_idx tracks the idx of the current cross-attn layer as we forward pass.
        # It's only going to be used if encoder_hidden_states contains a list of different tensors
        # to be cross-attended to.
        cross_attn_layer_idx = 0

        for i, (block, layer_past) in enumerate(zip(self.transformer.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Cross-attend first if not self.cross_attn_final_layer.
            if (
                not isinstance(self.cross_attn_layers[i], torch.nn.Identity)
                and not self.cross_attn_final_layer
            ):
                cross_attn_outputs = self.cross_attn_forward(
                    hidden_states=hidden_states,
                    layer_idx=i,
                    cross_attn_layer_idx=cross_attn_layer_idx,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                cross_attn_layer_idx += 1

                hidden_states = cross_attn_outputs[0]

            # Proceed to standard self-attn layer with inputs modified byt the cross-attn layer.
            # Applied previously if available in this layer.
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    None,
                    None,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]

            if use_cache:
                presents.append(outputs[1])

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Cross-attend after self-attn if self.cross_attn_final_layer.
            if (
                not isinstance(self.cross_attn_layers[i], torch.nn.Identity)
                and self.cross_attn_final_layer
            ):
                cross_attn_outputs = self.cross_attn_forward(
                    hidden_states=hidden_states,
                    layer_idx=i,
                    cross_attn_layer_idx=cross_attn_layer_idx,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                cross_attn_layer_idx += 1

                hidden_states = cross_attn_outputs[0]

        hidden_states = self.transformer.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (
                logits,
                presents,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            )
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def cross_attn_forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_idx: int,
        head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_layer_idx: Optional[
            int
        ] = -1,  # This is only used if encoder_hidden_states is a list of tensors to cross-attend to.
        encoder_hidden_states: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        encoder_attention_mask: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
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

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            cross_attn_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.cross_attn_layers[layer_idx]),
                hidden_states,
                None,
                head_mask[layer_idx],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            cross_attn_outputs = self.cross_attn_layers[layer_idx](
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        if self.cross_attn_skip_connections:
            return [hidden_states + cross_attn_outputs[0]]

        return cross_attn_outputs
