import os
import torch
import transformers
from transformers import TrainerCallback
from peft import set_peft_model_state_dict
from dssk.utils.scripting import LoggingCallback


class LoadBestPeftModelCallback(TrainerCallback):
    # Adpted from: https://github.com/bigcode-project/starcoder/blob/main/finetune/finetune.py
    def on_train_end(self, args, state, control, **kwargs):
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control


class WriteFinalStateCallback(TrainerCallback):
    # Save the trainer state in the root output folder,
    # to avoid having to guess which is the final checkpoint folder
    def on_train_end(self, args, state, control, **kwargs):
        state.save_to_json(os.path.join(args.output_dir, "final_trainer_state.json"))
        return control


class CustomTrainer(transformers.Trainer):
    """
    Custom trainer class for training to force label shifting for causal LM when label smoothing is used.
    Copied from research-dssk, but removing the missing grad in the encoder.
    """

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
    model,
    tokenizer,
    args,
    training_data,
    validation_data,
    maximum_input_length,
    wandb_entity_name=None,
    wandb_project_name=None,
    wandb_run_name=None,
    wandb_log_grads=False,
    cross_trainer=False,
):
    """Intanstiates Trainer object."""

    is_first_process = True
    if torch.distributed.is_initialized():
        is_first_process = torch.distributed.get_rank() == 0

    if is_first_process and wandb_entity_name is not None and wandb_project_name is not None:
        import wandb

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
            LoggingCallback(log_grads=wandb_log_grads),
        ]
    else:
        logging_callbacks = []

    if cross_trainer:
        from dssk.data.cross_attn.datasets_loader import Collator as CrossCollator

        trainer = CustomTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            data_collator=CrossCollator(
                tokenizer,
                maximum_length=maximum_input_length,
            ),
            train_dataset=training_data,
            eval_dataset=validation_data,
            callbacks=logging_callbacks + [WriteFinalStateCallback],
        )
    else:
        trainer = transformers.Trainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
            train_dataset=training_data,
            eval_dataset=validation_data,
            callbacks=logging_callbacks + [WriteFinalStateCallback],
        )

    return trainer
