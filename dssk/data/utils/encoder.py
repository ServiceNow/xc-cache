import os
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List


def remove_padding(padded_embeddings: torch.Tensor, padding_att_mask: torch.Tensor) -> List[float]:
    """Remove padding outputs from embeddings.

    Args:
        padded_embeddings (torch.Tensor): Embeddings with right padding outputs.
        padding_att_mask (torch.Tensor): Padding masks.

    Returns:
        List[float]: List of unpadded embedding tensors converted to nested python lists.
    """
    output_embedding_list = []
    for i in range(padding_att_mask.size(0)):
        non_padding_idx = padding_att_mask[i,].sum().item()
        non_padding_ids = padded_embeddings[i, :non_padding_idx, :]
        output_embedding_list.append(non_padding_ids.tolist())

    return output_embedding_list


class Encoder:
    """
    Please make sure a child process creation method is set to spawn
    in right multiprocessing library.

    The datasets.map uses multiprocess not multiprocessing.
    So, it must be multiprocess.set_start_method('spawn')
    """

    # NOTE: this implementation assumes available gpus indexes start from zero
    #   and go till the max_rank, so one device per rank
    # TODO: implement arbitrary cuda device list per rank
    def __init__(self, model_name: str, maximum_length: int, num_proc: int = 1) -> None:
        # this part is called in main process, so we do not initialize any model here
        super().__init__()

        # device_type will be set in the child process
        # as at least with the current versions of cuda and pytorch
        # any call which results with initialization of cuda in pytorch in the main process
        # will result in the cuda reinit error in the child process
        # if the start method is not set to spawn in multiprocess or multiprocessing libraries
        self.device_type = None
        self.device = None

        self.num_proc = num_proc
        self.model_name = model_name
        self.maximum_length = maximum_length
        self.encoder = None
        self.tokenizer = None
        self.parent_pid = os.getpid()
        self.worker_pid = None
        self.rank = None

    def _init(self, rank):
        # If self.num_proc > 1 this function must be called only in child process
        # also, it must be always called from the same child process
        if self.worker_pid is None:
            self.worker_pid = os.getpid()
        else:
            assert self.worker_pid == os.getpid()
        if self.num_proc > 1:
            assert self.worker_pid != self.parent_pid
            assert rank is not None
        elif rank is None:
            rank = 0

        # the function must be called for the same rank
        if self.rank is None:
            self.rank = rank
        else:
            assert self.rank == rank

        if self.encoder is not None:
            assert self.tokenizer is not None
            return
        assert self.tokenizer is None

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device_type == "cpu":
            device_map = {"": "cpu"}
            self.device = "cpu"
        else:
            assert self.device_type == "cuda"
            device_map = {"": rank}
            self.device = f"cuda:{rank}"


        if self.encoder is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            pad_token = self.tokenizer.pad_token
            if pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})

            self.encoder = AutoModel.from_pretrained(
                self.model_name,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                device_map=device_map,
            ).eval()

            self.encoder.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.no_grad()
    def encode(self, input_sentences: List[str], rank):
        # encode is called from child process, so we can safelly initialize model here
        self._init(rank)

        tokenized_outputs = self.tokenizer(input_sentences, padding=True, return_tensors="pt")
        input_ids = tokenized_outputs["input_ids"]
        att_mask = tokenized_outputs["attention_mask"]

        try:
            # Try to embed on device first.
            embedding = self.encoder( # We truncate long sequences to self.maximum_length
                input_ids=input_ids[:, :self.maximum_length].to(self.device),
                attention_mask=att_mask[:, :self.maximum_length].to(self.device),
                ).last_hidden_state.detach().detach().cpu()
        except:
            # In case of any error, we try again on cpu
            self.encoder = self.encoder.to("cpu")
            embedding = self.encoder( # We truncate long sequences to self.maximum_length
                input_ids=input_ids[:, :self.maximum_length].to("cpu"),
                attention_mask=att_mask[:, :self.maximum_length].to("cpu"),
                ).last_hidden_state.detach().detach()
            self.encoder = self.encoder.to(self.device)

        return remove_padding(embedding, att_mask)
