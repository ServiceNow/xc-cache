from calflops import calculate_flops


def estimate_flops_macs_params(model, gen_kwargs, forward_mode="generate"):
    total_flops, total_macs, total_params = calculate_flops(
        model=model, output_as_string=False, forward_mode=forward_mode, kwargs=gen_kwargs
    )

    return total_flops, total_macs, total_params
