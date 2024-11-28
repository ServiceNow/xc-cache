from typing import NamedTuple, Optional
import argparse
import os
import subprocess
import random
import numpy as np
import openai
import torch
from transformers.integrations import WandbCallback
from xc_cache.inference.cross_attn.inference_utils import Evaluator


# ******** Command line arguments ********


def parse_bool_flag(s: str) -> bool:
    """Interpret a string (from a command line arguments) as a boolean value

    Args:
        s (str): Input arg string.

    Returns:
        bool: _description_
    """
    _FALSY_STRINGS = {"off", "false", "no", "0"}
    _TRUTHY_STRINGS = {"on", "true", "yes", "1"}
    if s.lower() in _FALSY_STRINGS:
        return False
    elif s.lower() in _TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag")


# ******** torch.distributed, deepspeed etc. ********


class LocalRankAndWorldSize(NamedTuple):
    local_rank: int
    world_size: int


def get_local_rank_and_world_size() -> LocalRankAndWorldSize:
    """Source of truth for rank and world size
    This function can be used whether `torch.distributed` is initialized or not.
    If it *is* initialized, then this function's returned values should agree
    with `torch.distributed`'s own values.

    Our current convention is that the environment variables LOCAL_RANK and
    WORLD_SIZE are the underlying source of truth, which should be properly
    set automatically by torchrun/deepspeed/etc.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert 0 <= local_rank < world_size
    if torch.distributed.is_initialized():
        assert torch.distributed.get_rank() == local_rank
        assert torch.distributed.get_world_size() == world_size
    return LocalRankAndWorldSize(local_rank=local_rank, world_size=world_size)


def print_rank_0(logger, message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger(message)
    else:
        logger(message)


# ******** Compute environment ********


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds.

    Args:
        seed (int, optional): Defaults to 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed as per
    # https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    os.environ["PYTHONHASHSEED"] = str(seed)


# ******** Secrets/tokens ********


def get_hf_token() -> str:
    """Personal HuggingFace access token

    You must create the file `.hf_token` at the root of this repo, with
    a personal HuggingFace access token as its content.

    You can generate such a token at https://huggingface.co/settings/tokens .

    """
    try:
        with open(".hf_token", "rt") as f:
            return f.readline().strip()
    except FileNotFoundError as e:
        e.add_note(get_hf_token.__doc__)
        raise


def initialize_openai_token() -> None:
    """Personal OpenAI access token

    You must either create the file `.openai_token` at the root of this repo,
    with a personal OpenAI API key as its content,
    or you must manually set `openai.api_key` to your OpenAI API key value.

    """
    if openai.api_key is None:
        try:
            with open(".openai_token", "rt") as f:
                token = f.readline().strip()
            openai.api_key = token
        except FileNotFoundError as e:
            e.add_note(initialize_openai_token.__doc__)
            raise


class LoggingCallback(WandbCallback):
    """
    Overrigding WandbCallback to optionally turn off gradient logging,
    and carry out custom evaluation.
    """

    def __init__(
        self,
        log_grads: bool,
        do_extra_evals: bool = False,
        max_sample_size: Optional[int] = None,
    ):
        super().__init__()

        self.log_grads = log_grads
        # max_sample_size controls the number of generations. Set to None to use all validation data.
        self.evaluator = Evaluator(max_sample_size=max_sample_size)
        self.do_extra_evals = do_extra_evals

    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)
        _watch_model = "all" if self.log_grads else None
        self._wandb.watch(model, log=_watch_model, log_freq=max(100, args.logging_steps))
        if self.do_extra_evals:
            # Create table to log predictions.
            self.text_table = self._wandb.Table(
                columns=[
                    "step",
                    "eval loss",
                    "decoder inputs",
                    "generated answer",
                    "target answer",
                ]
            )

    def on_evaluate(self, args, state, control, **kwargs):
        print(kwargs.get("metrics"))
        if self.do_extra_evals:
            data_loader = kwargs.get("eval_dataloader")
            data_loader.collate_fn.return_generation_related_fields_(True)
            additional_metrics, decoder_inputs, predictions, raw_answers = self.evaluator(
                kwargs.get("model"),
                kwargs.get("eval_dataloader"),
                return_predictions_and_answers=True,
            )
            if self._wandb is None:
                return
            if not self._initialized:
                self.setup(args, state, kwargs.get("model"))
            if state.is_world_process_zero:
                print(additional_metrics)
                self._wandb.log(additional_metrics, step=state.global_step)
                self._log_generation_data(
                    decoder_inputs,
                    predictions,
                    raw_answers,
                    state.global_step,
                    kwargs.get("metrics")["eval_loss"],
                )
            data_loader.collate_fn.return_generation_related_fields_(False)

    def _log_generation_data(
        self, decoder_inputs, predictions, raw_answers, step, loss, max_n_added_rows=20
    ):
        # List[List[str]] to List[str]
        raw_answers_list = [x[0] for x in raw_answers]

        for i in range(min(len(decoder_inputs), max_n_added_rows)):
            self.text_table.add_data(
                step, loss, decoder_inputs[i], predictions[i], raw_answers_list[i]
            )
            print(
                f"\nInput: {decoder_inputs[i]} - Prediction: {predictions[i]} - Answer: {raw_answers_list[i]}"
            )

        self._wandb.log({"generation_samples": self.text_table})


# ******** Git ********


def is_git_clean() -> str:
    return len(subprocess.check_output(["git", "status", "--porcelain=v1", "2>/dev/null"])) == 0


def get_git_sha(assert_git_clean: bool = True) -> str:
    if assert_git_clean:
        assert is_git_clean(), "Unclean git! Add and/or commit your changes and try again."
    raw = subprocess.check_output(["git", "rev-parse", "HEAD"])
    try:
        sha = raw.decode()
        assert sha[-1] == "\n"
        sha = sha[:-1]
        assert len(sha) == 40
        assert all(hex in "0123456789abcdef" for hex in sha)
    except AssertionError:
        raise RuntimeError(f"Call to git returned: {raw}")
    return sha
