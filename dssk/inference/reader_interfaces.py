import json
from typing import Any, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

from baselines.fid.src.t5_wrapper import FiDT5

from dssk.models.cross_attn.load_checkpoint import load_checkpoint
from dssk.inference.abstract_lm_interface import AbstractLMInterface


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
        if model_ckpt:
            assert not model_path
            model, self.tokenizer = load_checkpoint(model_ckpt)
            model.eval()
        else:
            assert model_path
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_path)

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

        # Handle max length
        if model_max_length is None:
            model_max_length = self.model.transformer.wpe.num_embeddings
        self.tokenizer.model_max_length = model_max_length - max_new_tokens

    def __call__(self, sample: dict[str, Any], **gen_args) -> dict[str, Any]:
        """
        Any extra keyword arguments will be sent to the model.
        If a keyword is used both here and in the constructor, the one here will be used.
        """
        args = self.default_gen_args.copy()
        args.update(gen_args)

        # Input features that will be self-attended to.
        features = self.tokenizer(
            [sample["self_input_text"]], return_tensors="pt", truncation=True
        ).to(self.model.device)

        # Handle context(s) to be cross-attended (CURRENTLY SUPPORTS UP TO ONE CONTEXT!)
        cross_input_texts = sample.get("cross_input_texts", [])
        if cross_input_texts:
            if len(cross_input_texts) > 1:
                raise NotImplementedError("I don't know yet how to handle multiple contexts!")
            if len(cross_input_texts[0]) == 0:
                raise ValueError("Context has no chunks.")
            if len(cross_input_texts[0]) > 1:
                raise NotImplementedError("I don't know yet how to handle multiple chunks!")

            # At this point, there should be a single context, and it is in cross_input_texts[0][0]
            context_fts = self.tokenizer(
                [cross_input_texts[0][0]], return_tensors="pt", truncation=True
            ).to(self.model.device)
            args |= {
                "context_ids": context_fts["input_ids"],
                "encoder_attention_mask": context_fts["attention_mask"],
            }

        # pad_token_id is a valid kwargs for StarCoder, but it may not work for other models
        output = self.model.generate(**features, pad_token_id=self.tokenizer.eos_token_id, **args)

        num_input_tokens = len(features["input_ids"][0])
        try:
            output_text = self.tokenizer.decode(output[0, num_input_tokens:])
            return {
                "output_text": output_text,
                "error": False,
                "error_msg": "",
            }
        # Only recover from RuntimeError, and not CUDA OutOfMemoryError, since the later ones often lead
        # to an unstable GPU state, which can causes more error further down the line.
        except RuntimeError as e:
            return {
                "output_text": "",
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
            sample["self_input_text"],
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
            output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return {
                "output_text": output_text,
                "error": False,
                "error_msg": "",
            }
        # Only recover from RuntimeError, and not CUDA OutOfMemoryError, since the later ones often lead
        # to an unstable GPU state, which can cause more error further down the line.
        except RuntimeError as e:
            return {
                "output_text": "",
                "error": True,
                "error_msg": str(e),
            }

    @property
    def end_token(self) -> str:
        return self.tokenizer.eos_token
