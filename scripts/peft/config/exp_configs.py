BASE_CONFIG = {
    "model_path": "gpt2",
    "lora_cfg": None,
    "context_size": 512,
    "train_batch_size": 16,
    "test_batch_size": 16,
    "skip_steps": 1,  # Training steps to accumulate gradients through prior to updating params. This will override the value in the deepspeed config, if deepspeed is used.
    "learning_rate": 6e-4,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_epsilon": 1e-6,
    "weight_decay": 5e-2,
    "max_grad_norm": 1.0,
    "fp16": False,
    "bf16": False,
    "load_in_8bit": False,
    "n_workers": 2,
    "model_type": "llama",  # Only relevant for cross-attn models. Set in the specific config below if using a gptbicode, tulu, or mistral variant.
    "gradient_checkpointing": True,
}

MODEL_CONFIGS = {
    "cross_encoder_llama_7b": {
        "model_path": None,
        "checkpoint_path": "data_rw/test_ckpt/llama-7b/checkpoint-20000",
        "context_size": 8192,
        "lora_cfg": {  # https://github.com/huggingface/peft/blob/8f63f565c6baa93de4bd57c21d38e0ce4868c519/src/peft/tuners/lora.py#L40
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.5,
            "target_modules": ["q_proj", "v_proj"],
        },
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "weight_decay": 1e-2,
        "adam_epsilon": 1e-8,
        "load_in_8bit": True,
    },
    "lora_llama_7b_chat_small": {
        "model_path": "meta-llama/Llama-2-7b-chat-hf",
        "context_size": 8192,
        "lora_cfg": {  # https://github.com/huggingface/peft/blob/8f63f565c6baa93de4bd57c21d38e0ce4868c519/src/peft/tuners/lora.py#L40
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.5,
            "target_modules": ["q_proj", "v_proj"],
        },
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "weight_decay": 1e-2,
        "adam_epsilon": 1e-8,
        "load_in_8bit": True,
    },
    "lora_llama_7b_chat_big": {
        "model_path": "meta-llama/Llama-2-7b-chat-hf",
        "context_size": 8192,
        "lora_cfg": {  # https://github.com/huggingface/peft/blob/8f63f565c6baa93de4bd57c21d38e0ce4868c519/src/peft/tuners/lora.py#L40
            "r": 360,
            "lora_alpha": 720,  # twice r
            "lora_dropout": 0.5,
            "target_modules": ["q_proj", "v_proj", "k_proj"],
        },
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "weight_decay": 1e-2,
        "adam_epsilon": 1e-8,
        "load_in_8bit": True,
    },
    "lora_llama_7b_small": {
        "model_path": "meta-llama/Llama-2-7b-hf",
        "context_size": 8192,
        "lora_cfg": {  # https://github.com/huggingface/peft/blob/8f63f565c6baa93de4bd57c21d38e0ce4868c519/src/peft/tuners/lora.py#L40
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.5,
            "target_modules": ["q_proj", "v_proj"],
        },
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "weight_decay": 1e-2,
        "adam_epsilon": 1e-8,
        "load_in_8bit": True,
    },
    "lora_llama_minitest": {
        "model_path": "meta-llama/Llama-2-7b-hf",
        "context_size": 8192,
        "lora_cfg": {  # https://github.com/huggingface/peft/blob/8f63f565c6baa93de4bd57c21d38e0ce4868c519/src/peft/tuners/lora.py#L40
            "r": 360,
            "lora_alpha": 720,  # twice r
            "lora_dropout": 0.5,
            "target_modules": ["q_proj", "v_proj", "k_proj"],
        },
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "weight_decay": 1e-2,
        "adam_epsilon": 1e-8,
        "load_in_8bit": True,
    },
    "lora_llama_7b_big": {
        "model_path": "meta-llama/Llama-2-7b-hf",
        "context_size": 8192,
        "lora_cfg": {  # https://github.com/huggingface/peft/blob/8f63f565c6baa93de4bd57c21d38e0ce4868c519/src/peft/tuners/lora.py#L40
            "r": 360,
            "lora_alpha": 720,  # twice r
            "lora_dropout": 0.5,
            "target_modules": ["q_proj", "v_proj", "k_proj"],
        },
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "weight_decay": 1e-2,
        "adam_epsilon": 1e-8,
        "load_in_8bit": True,
    },
    "lora_tulu_7b_small": {
        "model_path": "allenai/tulu-2-dpo-7b",
        "context_size": 8192,
        "lora_cfg": {  # https://github.com/huggingface/peft/blob/8f63f565c6baa93de4bd57c21d38e0ce4868c519/src/peft/tuners/lora.py#L40
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.5,
            "target_modules": ["q_proj", "v_proj"],
        },
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "weight_decay": 1e-2,
        "adam_epsilon": 1e-8,
        "load_in_8bit": True,
    },
    "lora_tulu_7b_big": {
        "model_path": "allenai/tulu-2-dpo-7b",
        "context_size": 8192,
        "lora_cfg": {  # https://github.com/huggingface/peft/blob/8f63f565c6baa93de4bd57c21d38e0ce4868c519/src/peft/tuners/lora.py#L40
            "r": 360,
            "lora_alpha": 720,  # twice r
            "lora_dropout": 0.5,
            "target_modules": ["q_proj", "v_proj", "k_proj"],
        },
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "learning_rate": 1e-4,  # [2e-4, 1e-4, 5e-5, 2e-5, 1e-5]:
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "weight_decay": 1e-2,
        "adam_epsilon": 1e-8,
        "load_in_8bit": True,
    },
}

EXP_GROUPS = {}

for k, v in MODEL_CONFIGS.items():
    cfg = dict(BASE_CONFIG)
    cfg.update(v)
    EXP_GROUPS[k] = [cfg]
