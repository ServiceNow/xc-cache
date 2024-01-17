BASE_CONFIG = {  # Base cfg with rough defaults. Values used for training must be set in MODEL_CONFIGS below.
    "model_path": "TabbyML/SantaCoder-1B",
    "data_subset": "all",  # Expects values in {"all", "msmarco", "hotpotqa", "squad_v2", "nq", "topiocqa"}
    "include_questions_on_contexts": True,  # Whether to prepend questions on contexts.
    "num_cross_attn_layers": 4,
    "cross_attn_layers_stride": 2,
    "cross_attn_shared_weights": True,
    "cross_attn_dropout_prob": 0.3,
    "cross_attn_final_layer": False,
    "cross_attn_shared_projections": False,
    "cross_attn_hidden_size": None,
    "cross_attn_num_attention_heads": None,
    "context_size": 2048,
    "train_batch_size": 16,
    "test_batch_size": 16,
    "skip_steps": 1,  # Training steps to accumulate gradients through prior to updating params. This will override the value in the deepspeed config, if deepspeed is used.
    "learning_rate": 6e-4,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.05,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_epsilon": 1e-6,
    "weight_decay": 5e-2,
    "max_grad_norm": 1.0,
    "fp16": True,
    "bf16": False,
    "gradient_checkpointing": True,
    "include_context_ids": True,
    "label_smoothing_factor": 0.0,
    "cross_attn_num_key_value_heads": None,  # Only used for llama variants.
    "cross_attn_attention_bias": False,  # Only used for llama variants.
    "model_type": "gptbigcode",  # Set in the specific config below if using a llama variant.
}

MODEL_CONFIGS = {
    "crossattn_santacoder": {
        "model_path": "TabbyML/SantaCoder-1B",
        "num_cross_attn_layers": 6,
        "cross_attn_layers_stride": 2,
        "cross_attn_shared_weights": True,
        "train_batch_size": 5,
        "test_batch_size": 5,
        "skip_steps": 8,
        "context_size": 2048,
        "learning_rate": 3e-4,
        "weight_decay": 5e-4,
        "lr_scheduler_type": "cosine",
        "include_context_ids": True,
    },
    "crossattn_starcoder": {
        "model_path": "/mnt/dssk/data_rw/starcoder",
        "num_cross_attn_layers": 8,
        "cross_attn_layers_stride": 4,
        "cross_attn_shared_weights": True,
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 16,
        "context_size": 2048,
        "include_context_ids": True,
    },
    "crossattn_starcoderbase-1b": {
        "model_path": "/mnt/dssk/data_rw/hf_models/starcoderbase-1b",
        "num_cross_attn_layers": 6,
        "cross_attn_layers_stride": 3,
        "cross_attn_shared_weights": True,
        "train_batch_size": 2,
        "test_batch_size": 2,
        "skip_steps": 1,
        "context_size": 2048,
        "learning_rate": 2e-4,
        "weight_decay": 5e-1,
        "lr_scheduler_type": "reduce_lr_on_plateau",
        "include_context_ids": True,
    },
    "crossattn_starcoderbase-3b": {
        "model_path": "/mnt/dssk/data_rw/hf_models/starcoderbase-3b",
        "num_cross_attn_layers": 6,
        "cross_attn_layers_stride": 6,
        "cross_attn_shared_weights": False,
        "train_batch_size": 4,
        "test_batch_size": 4,
        "skip_steps": 32,
        "context_size": 4096,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
        "lr_scheduler_type": "cosine",
        "include_context_ids": True,
        "cross_attn_dropout_prob": 0.4,
        "cross_attn_final_layer": True,
        "cross_attn_shared_projections": False,
        "cross_attn_hidden_size": 1024,
        "cross_attn_num_attention_heads": 16,
        "label_smoothing_factor": 0.0,
    },
    "crossattn_starcoderbase-7b": {
        "model_path": "/mnt/dssk/data_rw/hf_models/starcoderbase-7b",
        "num_cross_attn_layers": 8,
        "cross_attn_layers_stride": 4,
        "cross_attn_shared_weights": True,
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "context_size": 2048,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "lr_scheduler_type": "cosine",
        "include_context_ids": True,
        "cross_attn_dropout_prob": 0.4,
        "cross_attn_final_layer": True,
        "cross_attn_shared_projections": True,
        "cross_attn_hidden_size": 1024,
        "cross_attn_num_attention_heads": 8,
        "fp16": True,
        "bf16": False,
    },
    "crossattn_starchat-alpha": {
        "model_path": "HuggingFaceH4/starchat-alpha",
        "num_cross_attn_layers": 4,
        "cross_attn_layers_stride": 8,
        "cross_attn_shared_weights": False,
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "context_size": 4096,  # NOTE: Using 4096 here requires deepspeed (OOM). Without deepspeed, use 2048.
        "learning_rate": 2e-4,
        "weight_decay": 1e-2,
        "lr_scheduler_type": "cosine",
        "include_context_ids": True,
        "cross_attn_dropout_prob": 0.25,
        "cross_attn_final_layer": True,
        "cross_attn_shared_projections": True,
        "cross_attn_hidden_size": 2048,
        "cross_attn_num_attention_heads": 32,
        "label_smoothing_factor": 0.0,
        "gradient_checkpointing": True,
        "fp16": False,
        "bf16": True,
    },
    "crossattn_tulu-7b": {
        "model_path": "allenai/tulu-2-dpo-7b",
        "data_subset": "all",  # Expects values in {"all", "msmarco", "hotpotqa", "squad_v2", "nq", "topiocqa"}
        "num_cross_attn_layers": 4,
        "cross_attn_layers_stride": 6,
        "cross_attn_shared_weights": False,
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "context_size": 6144,
        "learning_rate": 2e-4,
        "weight_decay": 1e-3,
        "lr_scheduler_type": "cosine",
        "include_context_ids": True,
        "cross_attn_dropout_prob": 0.2,
        "cross_attn_final_layer": True,
        "cross_attn_shared_projections": False,
        "cross_attn_hidden_size": 1024,
        "cross_attn_num_attention_heads": 32,
        "label_smoothing_factor": 0.0,
        "gradient_checkpointing": True,
        "fp16": False,
        "bf16": True,
        "cross_attn_num_key_value_heads": None,
        "cross_attn_attention_bias": False,
        "model_type": "llama",
    },
    "crossattn_tulu-13b": {
        "model_path": "allenai/tulu-2-dpo-13b",
        "data_subset": "all",  # Expects values in {"all", "msmarco", "hotpotqa", "squad_v2", "nq", "topiocqa"}
        "num_cross_attn_layers": 4,
        "cross_attn_layers_stride": 6,
        "cross_attn_shared_weights": False,
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "context_size": 6144,
        "learning_rate": 2e-4,
        "weight_decay": 1e-3,
        "lr_scheduler_type": "cosine",
        "include_context_ids": True,
        "cross_attn_dropout_prob": 0.2,
        "cross_attn_final_layer": True,
        "cross_attn_shared_projections": False,
        "cross_attn_hidden_size": 1024,
        "cross_attn_num_attention_heads": 32,
        "label_smoothing_factor": 0.0,
        "gradient_checkpointing": True,
        "fp16": False,
        "bf16": True,
        "cross_attn_num_key_value_heads": None,
        "cross_attn_attention_bias": False,
        "model_type": "llama",
    },
    "crossattn_tulu-7b_2x8": {  # Ablation case with 2 cross-attn layers and a stride of 8.
        "model_path": "allenai/tulu-2-dpo-7b",
        "data_subset": "all",  # Expects values in {"all", "msmarco", "hotpotqa", "squad_v2", "nq", "topiocqa"}
        "num_cross_attn_layers": 2,
        "cross_attn_layers_stride": 8,
        "cross_attn_shared_weights": False,
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "context_size": 6144,
        "learning_rate": 2e-4,
        "weight_decay": 1e-3,
        "lr_scheduler_type": "cosine",
        "include_context_ids": True,
        "cross_attn_dropout_prob": 0.2,
        "cross_attn_final_layer": True,
        "cross_attn_shared_projections": False,
        "cross_attn_hidden_size": 1024,
        "cross_attn_num_attention_heads": 32,
        "label_smoothing_factor": 0.0,
        "gradient_checkpointing": True,
        "fp16": False,
        "bf16": True,
        "cross_attn_num_key_value_heads": None,
        "cross_attn_attention_bias": False,
        "model_type": "llama",
    },
    "crossattn_tulu-7b_8x2": {  # Ablation case with 8 cross-attn layers and a stride of 2.
        "model_path": "allenai/tulu-2-dpo-7b",
        "data_subset": "all",  # Expects values in {"all", "msmarco", "hotpotqa", "squad_v2", "nq", "topiocqa"}
        "num_cross_attn_layers": 8,
        "cross_attn_layers_stride": 2,
        "cross_attn_shared_weights": False,
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "context_size": 6144,
        "learning_rate": 2e-4,
        "weight_decay": 1e-3,
        "lr_scheduler_type": "cosine",
        "include_context_ids": True,
        "cross_attn_dropout_prob": 0.2,
        "cross_attn_final_layer": True,
        "cross_attn_shared_projections": False,
        "cross_attn_hidden_size": 1024,
        "cross_attn_num_attention_heads": 32,
        "label_smoothing_factor": 0.0,
        "gradient_checkpointing": True,
        "fp16": False,
        "bf16": True,
        "cross_attn_num_key_value_heads": None,
        "cross_attn_attention_bias": False,
        "model_type": "llama",
    },
    "crossattn_tulu-7b_8x3": {  # Ablation case with 8 cross-attn layers and a stride of 3.
        "model_path": "allenai/tulu-2-dpo-7b",
        "data_subset": "all",  # Expects values in {"all", "msmarco", "hotpotqa", "squad_v2", "nq", "topiocqa"}
        "num_cross_attn_layers": 8,
        "cross_attn_layers_stride": 3,
        "cross_attn_shared_weights": False,
        "train_batch_size": 1,
        "test_batch_size": 1,
        "skip_steps": 32,
        "context_size": 6144,
        "learning_rate": 2e-4,
        "weight_decay": 1e-3,
        "lr_scheduler_type": "cosine",
        "include_context_ids": True,
        "cross_attn_dropout_prob": 0.2,
        "cross_attn_final_layer": True,
        "cross_attn_shared_projections": False,
        "cross_attn_hidden_size": 1024,
        "cross_attn_num_attention_heads": 32,
        "label_smoothing_factor": 0.0,
        "gradient_checkpointing": True,
        "fp16": False,
        "bf16": True,
        "cross_attn_num_key_value_heads": None,
        "cross_attn_attention_bias": False,
        "model_type": "llama",
    },
}

EXP_GROUPS = {}

for k, v in MODEL_CONFIGS.items():
    cfg = dict(BASE_CONFIG)
    cfg.update(v)
    EXP_GROUPS[k] = [cfg]
