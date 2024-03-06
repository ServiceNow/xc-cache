from typing import Optional
import torch
from dssk.metrics.generation.metrics import F1, EM, Precision, Recall
import tqdm


class Evaluator:
    def __init__(
        self,
        max_sample_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """Creates instance of Evaluatos.

        Args:
            max_sample_size (Optional[int]): Optional maximum number of samples for a rank to process.
            device (Optional[str]): Optional device str onto which move the data.
        """
        self.f1_evaluator = F1("")
        self.em_evaluator = EM("")
        self.precision_evaluator = Precision("")
        self.recall_evaluator = Recall("")
        self.max_sample_size = max_sample_size
        self.device = device

    def __call__(
        self,
        model,
        data_loader,
        return_predictions_and_answers=False,
        max_new_tokens=30,
        **generate_kwargs,
    ) -> float:
        """Compute evaluation metrics during training.
        A field called 'raw_answer' is expected to be populated in batches yielded by data_loader.
        It should contain a list of lists of str corresponding to answers to questions given contexts.
        """

        predictions = []
        raw_answers = []
        decoder_inputs = []

        for inputs in tqdm.tqdm(data_loader, desc="QA evaluation", total=len(data_loader)):
            # Drop labels if present.
            inputs.pop("labels", None)

            # Replace ids by the ones that do not contain the answer.
            no_answer_ids = inputs.pop("no_answer_input_ids")
            no_answer_att_mask = inputs.pop("no_answer_attention_mask")
            inputs["input_ids"] = no_answer_ids
            inputs["attention_mask"] = no_answer_att_mask

            raw_answers.extend(inputs.pop("raw_answer"))

            # Ensure data are on the correct device if self.device is not None.
            if self.device is not None:
                on_device_batch = {}
                for k, v in inputs.items():
                    try:
                        on_device_batch[k] = v.to(self.device)
                    except AttributeError:
                        # Non-tensor values (e.g., List[str]) can't be moved to a device
                        # so they are kept as-is.
                        on_device_batch[k] = v

                inputs = on_device_batch

            output = model.generate(
                **inputs,
                pad_token_id=data_loader.collate_fn.tokenizer.bos_token_id,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )

            generated_answers = []
            for i in range(output.size(0)):
                num_input_tokens = len(inputs["input_ids"][i])
                generated_answers.append(
                    data_loader.collate_fn.tokenizer.decode(
                        output[i, num_input_tokens:], skip_special_tokens=True
                    )
                )
                if return_predictions_and_answers:
                    decoder_inputs.append(
                        data_loader.collate_fn.tokenizer.decode(
                            inputs["input_ids"][i], skip_special_tokens=True
                        )
                    )

            predictions.extend(generated_answers)

            if self.max_sample_size is not None and len(predictions) >= self.max_sample_size:
                break

        if torch.distributed.is_initialized():
            # gather predictions and targets from other processes
            gather_objects = [predictions, raw_answers]
            # torch.cuda.device_count() upper bounds local world size.
            gathered_outputs = [None for _ in range(torch.cuda.device_count())]
            torch.distributed.gather_object(
                gather_objects,
                gathered_outputs if torch.distributed.get_rank() == 0 else None,
                dst=0,
            )

            if torch.distributed.get_rank() == 0:
                gathered_predictions, gathered_raw_answers = [], []
                for rank_output in gathered_outputs:
                    if rank_output is not None:
                        gathered_predictions.extend(rank_output[0])
                        gathered_raw_answers.extend(rank_output[1])

                predictions, raw_answers = gathered_predictions, gathered_raw_answers

        f1 = self.f1_evaluator(predictions, raw_answers)["f1"]
        em = self.em_evaluator(predictions, raw_answers)["em"]
        precision = self.precision_evaluator(predictions, raw_answers)["precision"]
        recall = self.recall_evaluator(predictions, raw_answers)["recall"]

        metrics = {
            "eval/f1": f1,
            "eval/em": em,
            "eval/precision": precision,
            "eval/recall": recall,
        }

        if return_predictions_and_answers:
            return metrics, decoder_inputs, predictions, raw_answers

        return metrics
