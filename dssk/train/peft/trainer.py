import os
import torch
import transformers
from typing import Optional
from transformers import TrainerCallback
from peft import set_peft_model_state_dict
from dssk.utils.scripting import LoggingCallback
from dssk.train.cross_attn.trainer import CustomTrainer
from dssk.data.cross_attn.datasets_loader import Collator as CrossCollator


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
    cross_trainer_do_extra_evals: bool = False,
    cross_trainer_generation_eval_max_sample_size: Optional[int] = None,
):
    """Intanstiates Trainer object."""

    if wandb_entity_name is not None and wandb_project_name is not None:
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
            LoggingCallback(
                log_grads=wandb_log_grads,
                do_extra_evals=cross_trainer_do_extra_evals and cross_trainer,
                max_sample_size=cross_trainer_generation_eval_max_sample_size,
            ),
        ]
    else:
        logging_callbacks = []

    if cross_trainer:

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
