import torch
from typing import List, Dict, Optional
from dssk.data.utils.encoder import Encoder


class WikipediaPreProcessor:
    def __init__(self, encoder: Encoder, field_name_prefix: Optional[str] = "") -> None:
        self.encoder = encoder
        self.field_name_prefix = field_name_prefix

    def __call__(self, examples: Dict[str, List], rank: int) -> Dict[str, torch.Tensor]:
        title_str = examples["title"]
        text_str = examples["text"]
        title_embedding = self.encoder.encode(title_str, rank)
        text_embedding = self.encoder.encode(text_str, rank)
        examples.update(
            {
                f"{self.field_name_prefix}title_embedding": title_embedding,
                f"{self.field_name_prefix}text_embedding": text_embedding,
            }
        )

        return examples


class SquadV2PreProcessor:
    def __init__(
        self, encoder: Encoder, context_length: int, return_context_embedding_only: bool = False
    ) -> None:
        self.encoder = encoder
        self.context_length = context_length
        self.return_context_embedding_only = return_context_embedding_only

    def __call__(self, examples: Dict[str, List], rank: int) -> Dict[str, torch.Tensor]:
        context_str = [f"context: {row}" for row in examples["context"]]

        context_embedding = self.encoder.encode(context_str, rank)

        if self.return_context_embedding_only:
            return {"encoder_hidden_states": context_embedding}

        question_str = [row for row in examples["question"]]

        answer_str = []
        for row in examples["answers"]:
            try:
                answer = row["text"][0]
            except IndexError:
                answer = ""

            answer_str.append(answer)

        input_str = [f"question: {q}\nanswer: {a}" for (q, a) in zip(question_str, answer_str)]

        input_ids = self.encoder.tokenizer(
            input_str,
            max_length=self.context_length,
            truncation=True,
        )["input_ids"]

        return {"input_ids": input_ids, "encoder_hidden_states": context_embedding}


class TrainDataPreProcessor:
    # TODO: substitute/use dssk.data.text_formats
    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder

    def __call__(self, examples: Dict[str, List], rank: int) -> Dict[str, torch.Tensor]:
        # rank is the process id. E.g., a 4-GPU processing job would have ranks in [0:4].
        # Check dssk.data.utils.encoders.Encoder

        context_str = [f"context: {row}" for row in examples["context"]]
        context_embedding = self.encoder.encode(context_str, rank)
        question_str = [row for row in examples["question"]]
        answer_str = [row for row in examples["answer"]]

        return {
            "context": context_str,
            "question": question_str,
            "answer": answer_str,
            "encoder_hidden_states": context_embedding,
        }
