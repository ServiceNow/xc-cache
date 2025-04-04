from argparse import ArgumentParser
from transformers import Trainer, PreTrainedTokenizerFast, PreTrainedModel
from datasets import Dataset
from xc_cache.data.cross_attn.datasets_loader import Collator
from typing import Optional


class CustomTrainer(Trainer):
    """Custom trainer class for training to force label shifting for causal LM when label smoothing is used."""

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # Drop fields related to generationin case those are in the batch since they're not used here.
        inputs.pop("raw_answer", None)
        inputs.pop("no_answer_input_ids", None)
        inputs.pop("no_answer_attention_mask", None)

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            # This is what we change wrt to the original compute_loss function.
            # We set shift_labels=True since our custom models doesn't exist
            # in the set of causal LM models within HF.
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def get_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    maximum_input_length: int,
    args: ArgumentParser,
    training_data: Dataset,
    validation_data: Dataset,
    wandb_entity_name: Optional[str] = None,
    wandb_project_name: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_log_grads: Optional[bool] = None,
    do_extra_evals: bool = False,
    generation_eval_max_sample_size: Optional[int] = None,
) -> Trainer:
    """Intanstiates Trainer object.

    Args:
        model (PreTrainedModel): Model to be trained.
        tokenizer (PreTrainedTokenizerFast): Model's tokenizer.
        maximum_input_length (int): Maximum length of input id sequences.
        args (ArgumentParser): Command line arguments.
        training_data (Dataset): Training dataset.
        validation_data (Dataset): Validation dataset.
        wandb_entity_name (Optional[str], optional): Defaults to None.
        wandb_project_name (Optional[str], optional): Defaults to None.
        wandb_run_name (Optional[str]): Defaults to None.
        wandb_log_grads (Optional[bool]): Wheter to log gradients on wandb. Defaults to None.
        do_extra_evals (Optional[bool]): whether to run extra evaluations within logging.
        generation_eval_max_sample_size (Optional[int]): Maximum sample size for evaluation that involves generation.

    Returns:
        Trainer: Configured trainer.
    """

    if wandb_entity_name is not None and wandb_project_name is not None:
        import wandb
        from xc_cache.utils.scripting import LoggingCallback

        try:
            wandb.init(
                name=wandb_run_name,
                entity=wandb_entity_name,
                project=wandb_project_name,
            )
        except wandb.AuthenticationError:
            with open("keys", "r") as f:
                key = f.read()
            wandb.login(key=key)
            wandb.init(
                name=wandb_run_name,
                entity=wandb_entity_name,
                project=wandb_project_name,
            )

        logging_callbacks = [
            LoggingCallback(
                log_grads=wandb_log_grads,
                do_extra_evals=do_extra_evals,
                max_sample_size=generation_eval_max_sample_size,
            ),
        ]
    else:
        logging_callbacks = []

    trainer = CustomTrainer(
        model=model,
        args=args,
        data_collator=Collator(tokenizer, maximum_length=maximum_input_length),
        train_dataset=training_data,
        eval_dataset=validation_data,
        callbacks=logging_callbacks,
    )

    return trainer
