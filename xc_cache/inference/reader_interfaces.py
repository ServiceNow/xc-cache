import os
import json
from typing import Any, Optional

import torch
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, PreTrainedModel

from baselines.fid.src.t5_wrapper import FiDT5

from xc_cache.models.cross_attn.load_checkpoint import load_checkpoint
from xc_cache.inference.abstract_lm_interface import AbstractLMInterface

from safetensors.torch import load_file as load_safetensors


def _setup_peft_model(
    model: PreTrainedModel, peft_config: dict[str, Any], model_peft_ckpt: str
) -> peft.PeftModel:
    with open(peft_config, "rt") as fp:
        peft_config = json.load(fp)
    peft_config = peft.LoraConfig(task_type=peft.TaskType.CAUSAL_LM, **peft_config)

    for name, param in model.named_parameters():
        # upcast LM head and layernorms
        if any([k in name for k in ["lm_head", "wte", "ln_"]]):
            param.data = param.data.to(torch.float32)

    model = peft.get_peft_model(model, peft_config)

    if model_peft_ckpt.endswith(".safetensors"):
        adapters_weights = load_safetensors(model_peft_ckpt, device="cpu")
    else:
        adapters_weights = torch.load(model_peft_ckpt, map_location="cpu")

    _ = peft.set_peft_model_state_dict(model, adapters_weights)

    # Make sure the extra model parts from PEFT are also in evaluation mode
    return model.eval()


class CrossAttnInterface(AbstractLMInterface):
    """
    An interface with any HuggingFace model which uses AutoTokenizer and AutoModelForCausalLM.
    Extra keyword arguments will be sent to the model at every call.
    """

    def __init__(
        self,
        *,
        max_new_tokens: int,
        cache_path: str,
        model_max_length: Optional[int] = None,
        model_path: Optional[str] = None,
        model_ckpt: Optional[str] = None,
        peft_config: Optional[str] = None,
        model_peft_ckpt: Optional[str] = None,
        ds_config: Optional[str] = None,
        to_device: Optional[str] = None,
        default_gen_args: Optional[dict[str, Any]] = None,
        **kwargs,  # Ignored
    ):
        self.default_gen_args = default_gen_args.copy() if default_gen_args is not None else {}
        if max_new_tokens in self.default_gen_args:
            assert self.default_gen_args["max_new_tokens"] == max_new_tokens
        else:
            self.default_gen_args["max_new_tokens"] = max_new_tokens

        # First half of deepspeed configuration (continued after model loading).
        if ds_config is not None:
            with open(ds_config, "rt") as fp:
                ds_config = json.load(fp)
            import deepspeed
            from transformers.deepspeed import HfDeepSpeedConfig

            deepspeed.init_distributed("nccl")
            # Next line does magic so that from_pretrained instantiates directly on gpus.
            # NOTE: This MUST happen before model loading!
            _ = HfDeepSpeedConfig(ds_config)

        # Model loading
        self.model_ckpt = model_ckpt
        if model_ckpt:
            assert not model_path
            model, self.tokenizer = load_checkpoint(model_ckpt)
            model.eval()
        else:
            assert model_path
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_path)

        self.model_peft_ckpt = model_peft_ckpt
        if peft_config is not None:
            assert self.model_peft_ckpt is not None
            model = _setup_peft_model(model, peft_config, self.model_peft_ckpt)
        else:
            assert self.model_peft_ckpt is None

        # Second half of deepspeed configuration.
        if ds_config is not None:
            assert to_device is None
            ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
            ds_engine.module.eval()
            self.model = ds_engine.module
        else:
            self.model = model
            self.model.eval()
            if to_device is not None:
                if "LOCAL_RANK" in to_device:
                    to_device = to_device.replace("LOCAL_RANK", os.environ.get("LOCAL_RANK", "0"))
                self.model.to(to_device)

        # Handle max length
        if model_max_length is None:
            if hasattr(self.model.config, "max_len"):
                model_max_length = self.model.config.max_len
            elif hasattr(self.model.config, "max_position_embeddings"):
                model_max_length = self.model.config.max_position_embeddings
            elif "wpe" in self.model.transformer:
                model_max_length = self.model.transformer.wpe.num_embeddings
            else:
                raise ValueError("Cannot automatically infer model_max_length.")
        self.tokenizer.model_max_length = model_max_length

        # Handle input format options
        # Cross-attn models trained before this field was added to the config were trained with
        # questions on contexts, hence the default to True.
        self.include_questions_on_contexts = getattr(
            self.model.config, "include_questions_on_contexts", True
        )
        # Cross-attn models trained before this field was added to the config were trained with
        # concatenated contexts, hence the default to False.
        self.return_context_list = getattr(self.model.config, "chunked_contexts", False)

    @property
    def model_info(self) -> dict[str, Any]:
        # See docstring in AbstractLMInterface.model_info
        return {
            "class_name": self.model.__class__.__name__,
            "name_or_path": self.model.name_or_path,
            "model_ckpt": self.model_ckpt,
            "model_peft_ckpt": self.model_peft_ckpt,
            "default_gen_args": self.default_gen_args,
        }

    def __call__(self, sample: dict[str, Any], **gen_args) -> dict[str, Any]:
        """
        Any extra keyword arguments will be sent to the model.
        If a keyword is used both here and in the constructor, the one here will be used.
        """
        args = self.default_gen_args.copy()
        args.update(gen_args)

        if "no_answer_self_input_str" in sample:
            input_str = sample["no_answer_self_input_str"]
        elif "self_input_str" in sample:
            input_str = sample["self_input_str"]
        else:
            input_str = sample["input_str"]

        # Input features that will be self-attended to.
        features = self.tokenizer([input_str], return_tensors="pt", truncation=True).to(
            self.model.device
        )

        # Drop <eos> token inserted by tokenizer from the decoder's inputs.
        if features["input_ids"][0, -1] == self.tokenizer.eos_token_id:
            features["input_ids"] = features["input_ids"][..., :-1]
            features["attention_mask"] = features["attention_mask"][..., :-1]

        # Decide whether to include the question at the start of the context.
        if self.include_questions_on_contexts:
            cross_input_key = "cross_input_str_with_question"
        else:
            cross_input_key = "cross_input_str"

        # Context features to be cross-attended to
        if cross_input_key in sample and sample[cross_input_key]:
            if isinstance(sample[cross_input_key], list):
                context_ids_list, encoder_attn_mask_list = [], []
                for context_str in sample[cross_input_key]:
                    context_fts = self.tokenizer(
                        [context_str], return_tensors="pt", truncation=True
                    ).to(self.model.device)
                    context_ids_list.append(context_fts["input_ids"])
                    encoder_attn_mask_list.append(context_fts["attention_mask"])
                args |= {
                    "context_input_ids": context_ids_list,
                    "encoder_attention_mask": encoder_attn_mask_list,
                }
            else:
                context_fts = self.tokenizer(
                    [sample["cross_input_str"]], return_tensors="pt", truncation=True
                ).to(self.model.device)
                args |= {
                    "context_input_ids": context_fts["input_ids"],
                    "encoder_attention_mask": context_fts["attention_mask"],
                }

        # pad_token_id is a valid kwargs for StarCoder, but it may not work for other models
        output = self.model.generate(**features, pad_token_id=self.tokenizer.bos_token_id, **args)

        num_input_tokens = len(features["input_ids"][0])
        try:
            output_text = self.tokenizer.decode(
                output[0, num_input_tokens:], skip_special_tokens=True
            )
            return {
                "answer_pred": output_text,
                "error": False,
                "error_msg": "",
            }
        # Only recover from RuntimeError, and not CUDA OutOfMemoryError, since the later ones often lead
        # to an unstable GPU state, which can causes more error further down the line.
        except RuntimeError as e:
            return {
                "answer_pred": "",
                "error": True,
                "error_msg": str(e),
            }

    @property
    def end_token(self) -> str:
        return self.tokenizer.eos_token


class FiDInterface(AbstractLMInterface):
    """
    An interface for FiD model.
    Extra keyword arguments will be sent to the model at every call.
    """

    def __init__(
        self,
        *,
        max_new_tokens: int,
        cache_path: str,
        model_path: str,
        model_max_length: Optional[int] = None,  # opt.text_maxlegth
        ds_config: Optional[str] = None,
        to_device: Optional[str] = None,
        default_gen_args: Optional[dict[str, Any]] = None,
        **kwargs,  # Ignored
    ):
        self.default_gen_args = default_gen_args.copy() if default_gen_args is not None else {}
        if max_new_tokens in self.default_gen_args:
            assert self.default_gen_args["max_length"] == max_new_tokens
        else:
            self.default_gen_args["max_length"] = max_new_tokens

        # First half of deepspeed configuration (continued after model loading).
        if ds_config is not None:
            with open(ds_config, "rt") as fp:
                ds_config = json.load(fp)
            import deepspeed
            from transformers.deepspeed import HfDeepSpeedConfig

            deepspeed.init_distributed("nccl")
            # Next line does magic so that from_pretrained instantiates directly on gpus.
            # NOTE: This MUST happen before model loading!
            _ = HfDeepSpeedConfig(ds_config)

        # Model loading
        model = FiDT5.from_pretrained(model_path, cache_dir=cache_path)

        # Handle input max length (per passage). set to 250 in original paper
        if model_max_length is None:
            model_max_length = int(model.config.d_model)

        # load tokenizer (same for all T5 sizes)
        self.tokenizer = T5Tokenizer.from_pretrained(
            "t5-base", model_max_length=model_max_length, return_dict=False
        )

        # Second half of deepspeed configuration.
        if ds_config is not None:
            assert to_device is None
            ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
            ds_engine.module.eval()
            self.model = ds_engine.module
        else:
            self.model = model
            self.model.eval()
            if to_device is not None:
                self.model.to(to_device)

    @property
    def model_info(self) -> dict[str, Any]:
        # See docstring in AbstractLMInterface.model_info
        return {
            "class_name": self.model.__class__.__name__,
            "name_or_path": self.model.name_or_path,
        }

    def __call__(self, sample: dict[str, Any], **gen_args) -> dict[str, Any]:
        """
        Any extra keyword arguments will be sent to the model.
        If a keyword is used both here and in the constructor, the one here will be used.
        """
        args = self.default_gen_args.copy()
        args.update(gen_args)

        # Tokenize list of passages
        # batch, because we have multiple passages per sample
        features = self.tokenizer.batch_encode_plus(
            sample["passages"],
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device)

        # FiD's generate() supports only batch generation. Need to inflate dimensions
        output = self.model.generate(
            input_ids=features["input_ids"][None],
            attention_mask=features["attention_mask"][None],
            **args,
        )

        try:
            # A single answer should be returned
            assert len(output) == 1
            answer_pred = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return {
                "answer_pred": answer_pred,
                "error": False,
                "error_msg": "",
            }
        # Only recover from RuntimeError, and not CUDA OutOfMemoryError, since the later ones often lead
        # to an unstable GPU state, which can cause more error further down the line.
        except RuntimeError as e:
            return {
                "answer_pred": "",
                "error": True,
                "error_msg": str(e),
            }

    @property
    def end_token(self) -> str:
        return self.tokenizer.eos_token
