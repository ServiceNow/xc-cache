from typing import Optional, List, Dict, Union, Any
import torch
from transformers import PreTrainedTokenizerFast
from dssk.metrics.generation.metrics import F1, EM, Precision, Recall
import tqdm


def send_data_to_device(
    data: Dict[str, Union[torch.Tensor, Any]], device: Union[str, torch.device]
) -> Dict[str, Union[torch.Tensor, Any]]:
    # Ensure data are on the correct device if self.device is not None.
    on_device_data = {}
    for k, v in data.items():
        try:
            on_device_data[k] = v.to(device)
        except AttributeError:
            # Non-tensor values (e.g., List[str]) can't be moved to a device
            # so they are kept as-is.
            on_device_data[k] = v

    return on_device_data


def process_inputs(
    *,
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizerFast,
    max_new_tokens: int,
    device: Union[str, torch.device],
    generate_kwargs: Dict[str, Any],
) -> List[List[str]]:
    """This function takes as input a batch of data as output by an instance of dssk.data.cross_attn.data_processors.Collator,
    and a model and tokenizer, and returns predictions from the model. Model inputs that yielded generations are also returned.
    """
    # Drop labels if present.
    inputs.pop("labels", None)
    datasets_list = inputs.pop("datasets", None)

    # Replace ids by the ones that do not contain the answer.
    # Unlike training, for evaluation, the decoder receives input ids
    # that do not contain the answer since we want to generate answers.
    no_answer_ids = inputs.pop("no_answer_input_ids")
    no_answer_att_mask = inputs.pop("no_answer_attention_mask")
    inputs["input_ids"] = no_answer_ids
    inputs["attention_mask"] = no_answer_att_mask

    raw_answers = [
        [tokenizer.decode(x, skip_special_tokens=True)]
        for x in inputs.pop("raw_answer_input_ids").cpu()
    ]

    # Ensure data are on the correct device if self.device is not None.
    inputs = send_data_to_device(data=inputs, device=device)

    with torch.no_grad():
        try:
            output = model.generate(
                **inputs,
                pad_token_id=tokenizer.bos_token_id,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
        except RuntimeError:
            Warning("Got RunTimeError when trying to generate. Will retry generation on cpu.")
            model = model.cpu()
            inputs = send_data_to_device(data=inputs, device="cpu")
            output = model.generate(
                **inputs,
                pad_token_id=tokenizer.bos_token_id,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
            model = model.to(device)

    predictions = []
    decoder_inputs = []
    for i in range(output.size(0)):
        num_input_tokens = len(inputs["input_ids"][i])
        predictions.append(
            tokenizer.decode(output[i, num_input_tokens:], skip_special_tokens=True)
        )
        decoder_inputs.append(tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True))
    return predictions, raw_answers, decoder_inputs, datasets_list


class Evaluator:
    def __init__(
        self,
        max_sample_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """Creates instance of Evaluator.

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

        model = model.eval()

        if self.device is None:
            self.device = next(model.parameters()).device

        predictions = []
        raw_answers = []
        decoder_inputs = []
        datasets_ids = []
        dataset_predictions = {}

        for inputs in tqdm.tqdm(data_loader, desc="QA evaluation", total=len(data_loader)):
            new_predictions, new_raw_answers, new_decoder_inputs, new_datasets_ids = (
                process_inputs(
                    model=model,
                    inputs=inputs,
                    tokenizer=data_loader.collate_fn.tokenizer,
                    max_new_tokens=max_new_tokens,
                    device=self.device,
                    generate_kwargs=generate_kwargs,
                )
            )
            predictions.extend(new_predictions)
            raw_answers.extend(new_raw_answers)
            decoder_inputs.extend(new_decoder_inputs)
            datasets_ids.extend(new_datasets_ids)

            if self.max_sample_size is not None and len(predictions) >= self.max_sample_size:
                break

        if torch.distributed.is_initialized():
            # gather predictions, targets, and dataset names from other processes
            gather_objects = [predictions, raw_answers, datasets_ids]
            # torch.cuda.device_count() upper bounds local world size.
            gathered_outputs = [None for _ in range(torch.cuda.device_count())]
            torch.distributed.gather_object(
                gather_objects,
                gathered_outputs if torch.distributed.get_rank() == 0 else None,
                dst=0,
            )

            if torch.distributed.get_rank() == 0:
                gathered_predictions, gathered_raw_answers, gathered_dataset_ids = [], [], []
                for rank_output in gathered_outputs:
                    if rank_output is not None:
                        gathered_predictions.extend(rank_output[0])
                        gathered_raw_answers.extend(rank_output[1])
                        gathered_dataset_ids.extend(rank_output[2])

                predictions, raw_answers, datasets_ids = (
                    gathered_predictions,
                    gathered_raw_answers,
                    gathered_dataset_ids,
                )

        # Split predictions and ground truth answer per dataset
        for i, dataset_name in enumerate(datasets_ids):
            if dataset_name not in dataset_predictions:
                dataset_predictions[dataset_name] = {
                    "predictions": [],
                    "raw_answers": [],
                }
            dataset_predictions[dataset_name]["predictions"].append(predictions[i])
            dataset_predictions[dataset_name]["raw_answers"].append(raw_answers[i])

        metrics = {}
        # Get results per dataset
        for k, v in dataset_predictions.items():
            f1 = self.f1_evaluator(v["predictions"], v["raw_answers"])["f1"]
            em = self.em_evaluator(v["predictions"], v["raw_answers"])["em"]
            precision = self.precision_evaluator(v["predictions"], v["raw_answers"])["precision"]
            recall = self.recall_evaluator(v["predictions"], v["raw_answers"])["recall"]

            metrics.update(
                {
                    f"eval/f1_{k}": f1,
                    f"eval/em_{k}": em,
                    f"eval/precision_{k}": precision,
                    f"eval/recall_{k}": recall,
                }
            )

        # Get overall results
        f1 = self.f1_evaluator(predictions, raw_answers)["f1"]
        em = self.em_evaluator(predictions, raw_answers)["em"]
        precision = self.precision_evaluator(predictions, raw_answers)["precision"]
        recall = self.recall_evaluator(predictions, raw_answers)["recall"]

        metrics.update(
            {
                "eval/f1_overall": f1,
                "eval/em_overall": em,
                "eval/precision_overall": precision,
                "eval/recall_overall": recall,
            }
        )

        if return_predictions_and_answers:
            return metrics, decoder_inputs, predictions, raw_answers

        return metrics
