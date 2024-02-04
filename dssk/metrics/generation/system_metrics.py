from calflops import calculate_flops


def estimate_flops_macs(model, tokenizer, batch_size, context_length, qa_length, **kwargs):
    # supports only decoder-only models
    total_flops, total_macs, _ = calculate_flops(
        model=model,
        input_shape=(batch_size, context_length + qa_length),
        transformer_tokenizer=tokenizer,
        output_as_string=False,
    )

    return total_flops, total_macs
