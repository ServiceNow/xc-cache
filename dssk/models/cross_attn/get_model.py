from typing import Optional, Union
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
from dssk.models.cross_attn.cross_attn_gptbigcode import CrossAttnGPTBigCode
from dssk.models.cross_attn.cross_attn_llama import CrossAttnLlama
from dssk.models import infer_model_type


def get_model(
    model_path: str,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    use_gradient_checkpointing: Optional[bool] = False,
    device: Optional[Union[str, torch.device]] = None,
    n_cross_attn_layers: Optional[int] = 0,
    cross_attn_layers_stride: Optional[int] = 1,
    cross_attn_shared_weights: Optional[bool] = True,
    cross_attn_dropout_prob: Optional[float] = 0.0,
    cross_attn_final_layer: Optional[bool] = False,
    cross_attn_shared_projections: Optional[bool] = False,
    cross_attn_hidden_size: Optional[int] = None,
    cross_attn_num_attention_heads: Optional[int] = None,
    randomly_initialize_decoder: Optional[bool] = False,
    model_type: Optional[str] = None,
    cross_attn_num_key_value_heads: Optional[int] = None,
    cross_attn_attention_bias: Optional[bool] = False,
    cache_dir: Optional[str] = None,
    max_len: int = -1,
) -> PreTrainedModel:
    """Helper function to get models..
    For models that are instances of GPTBigCodeForCausalLM, optionally
    add cross-attn layers to the loaded model.

    Args:
        model_path (str): Local path or huggingface hub id to model.
        bos_token_id (int):  BOS token id to be added to model config.
        eos_token_id (int):  EOS token id to be added to model config.
        pad_token_id (int):  Padding token id to be added to model config.
        use_gradient_checkpointing (Optional[bool]): Whether to enable gradient checkpointing. Defaults to False,
        device: Optional[Union[str, torch.device, None]]:  Device where to load the model. Defaults to None.
        n_cross_attn_layers (Optional[int], optional): Number of cross-attn layers to be added to the model. Defaults to 0.
        cross_attn_layers_stride (Optional[int]): Stride for adding cross-attn layers. Defaults to 1.
        cross_attn_shared_weights (Optional[bool]): Whether to share cross-attn parameters. Defaults to True.
        cross_attn_dropout_prob (Optional[float]): Dropout probability for cross-attn attention masks. Defaults to 0.0 (no dropout.)
        cross_attn_final_layer (Optional[bool]): Whether the last layer is a cross-attn layer. Defaults to False.
        cross_attn_shared_projections (Optional[bool]): Whether to share parameters for query and key projections. Defaults to False.
        cross_attn_hidden_size (Optional[int]): If None (default), will use the base decoder's hidden size.
        cross_attn_num_attention_heads (Optional[int]): If None (default), will use the base decoder's number of attn heads.
        randomly_initialize_decoder (Optional[bool]): Whether to randomly initialize the decoder. Defaults to False.
        model_type (Optional[str]): Which kind of model to instantiate. We currently support values in {"llama", "gpt_bigcode"}.
        cross_attn_num_key_value_heads (Optional[int]): Only used for Llama variations. If None (default), will use the base decoder's number of attn heads.
        cross_attn_attention_bias (Optional[bool]): Only used for Llama variations. Whether to train bias parameters.
        cache_dir (Optional[str]): Optional path to store hf files for pretrained models.
        max_len: (int): Optional value of the maximum model length to be added to the model cfg. Useful for inference. Default -1 means "unset".

    Returns:
        PreTrainedModel: Pre-trained model.
    """
    if model_type is None:
        model_type = infer_model_type(model_path)
    model_type = model_type.lower().replace("_", "").replace("-", "")
    if n_cross_attn_layers > 0:
        if model_type == "llama":
            model = CrossAttnLlama(
                model_path,
                n_cross_attn_layers=n_cross_attn_layers,
                cross_attn_layers_stride=cross_attn_layers_stride,
                cross_attn_shared_weights=cross_attn_shared_weights,
                cross_attn_dropout_prob=cross_attn_dropout_prob,
                cross_attn_final_layer=cross_attn_final_layer,
                cross_attn_shared_projections=cross_attn_shared_projections,
                cross_attn_hidden_size=cross_attn_hidden_size,
                cross_attn_num_attention_heads=cross_attn_num_attention_heads,
                cross_attn_num_key_value_heads=cross_attn_num_key_value_heads,
                cross_attn_attention_bias=cross_attn_attention_bias,
                randomly_initialize_decoder=randomly_initialize_decoder,
                cache_dir=cache_dir,
                max_len=max_len,
            )
        elif model_type == "gptbigcode":
            model = CrossAttnGPTBigCode(
                model_path,
                n_cross_attn_layers=n_cross_attn_layers,
                cross_attn_layers_stride=cross_attn_layers_stride,
                cross_attn_shared_weights=cross_attn_shared_weights,
                cross_attn_dropout_prob=cross_attn_dropout_prob,
                cross_attn_final_layer=cross_attn_final_layer,
                cross_attn_shared_projections=cross_attn_shared_projections,
                cross_attn_hidden_size=cross_attn_hidden_size,
                cross_attn_num_attention_heads=cross_attn_num_attention_heads,
                randomly_initialize_decoder=randomly_initialize_decoder,
                cache_dir=cache_dir,
                max_len=max_len,
            )
        else:
            raise ValueError(f"Got unsupported model_type {model_type}.")

        if device is not None:
            model = model.to(device)
        model.bos_token_id = bos_token_id
        model.eos_token_id = eos_token_id
        model = model.prepare_for_training(
            train_all_params=False,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    else:
        # This branch is mostly used to instantiate pre-processing models.
        # We then handle device allocation manually to avoid several processes
        # Loading models into ram simultaneously. device_map handles it if
        # device is specified.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            device_map={
                "": device or "cpu"
            },  # loads directly on device to avoid multiple processes loading models on ram.
        )

    # TODO: Why is `config` here and not above?
    model.config.pad_token_id = pad_token_id

    return model
