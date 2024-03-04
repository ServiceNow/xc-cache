import torch
import transformers
import peft

from dssk.models.cross_attn.load_checkpoint import load_checkpoint


def prepare_model_for_bf16_lora(model, use_gradient_checkpointing=False):
    for _, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    for name, param in model.named_parameters():
        # upcast LM head and layernorms
        if any([k in name for k in ["lm_head", "wte", "ln_"]]):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    return model


def get_model(model_path, lora_config, cache_dir, load_in_8bit, process_index=0):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, padding="max_length", truncation="max_length", use_auth_token=True
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        cache_dir=cache_dir,
        use_auth_token=True,
        load_in_8bit=load_in_8bit,
        device_map={"": process_index},
    )

    if load_in_8bit:
        model = peft.prepare_model_for_int8_training(model)
    elif lora_config:
        model = prepare_model_for_bf16_lora(model)

    if lora_config is not None:
        peft_config = peft.LoraConfig(task_type=peft.TaskType.CAUSAL_LM, **lora_config)
        model = peft.get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


def get_cross_model(ckp_path, lora_config, load_in_8bit):

    model, _ = load_checkpoint(ckp_path)

    if load_in_8bit:
        model = peft.prepare_model_for_int8_training(model)
    elif lora_config:
        model = prepare_model_for_bf16_lora(model)

    if lora_config is not None:
        peft_config = peft.LoraConfig(task_type=peft.TaskType.CAUSAL_LM, **lora_config)
        model = peft.get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model
