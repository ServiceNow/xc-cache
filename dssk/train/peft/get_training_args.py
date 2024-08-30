import os
import json
import io
import transformers


def get_training_args(exp_dict, args, savedir):
    if args.deepspeed is None or args.deepspeed.lower() == "none":
        deepspeed_config = None
    else:
        with io.open(args.deepspeed, "r", encoding="utf-8") as f:
            deepspeed_config = json.load(f)

        if "auto" not in deepspeed_config["gradient_accumulation_steps"]:
            # Override value with skip_steps from the cfg in exp_configs.py
            deepspeed_config["gradient_accumulation_steps"] = exp_dict["skip_steps"]

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=savedir,
        local_rank=args.local_rank,
        per_device_train_batch_size=exp_dict["train_batch_size"],
        per_device_eval_batch_size=exp_dict["test_batch_size"],
        max_steps=args.steps,
        learning_rate=exp_dict["learning_rate"],
        lr_scheduler_type=exp_dict["lr_scheduler_type"],
        warmup_ratio=exp_dict["warmup_ratio"],
        adam_beta1=exp_dict["adam_beta1"],
        adam_beta2=exp_dict["adam_beta2"],
        adam_epsilon=exp_dict["adam_epsilon"],
        weight_decay=exp_dict["weight_decay"],
        max_grad_norm=exp_dict["max_grad_norm"],
        gradient_accumulation_steps=exp_dict["skip_steps"],
        gradient_checkpointing=exp_dict["gradient_checkpointing"],
        fp16=exp_dict["fp16"],
        bf16=exp_dict["bf16"],
        logging_dir=os.path.join(savedir, "logs"),
        logging_strategy="steps",
        logging_steps=args.log_every,
        save_strategy="steps",
        save_steps=args.log_every,
        evaluation_strategy="steps",
        ddp_find_unused_parameters=False,
        deepspeed=deepspeed_config,
        push_to_hub=args.push_to_hub,
        load_best_model_at_end=False,
        remove_unused_columns=False,  # This needs to be False since our custom dataset passes fields that are processed in the collator.
        # save_total_limit=5,
    )

    return training_args
