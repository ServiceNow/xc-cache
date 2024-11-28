"""
Given a QA dataset and a method to answer the questions, this file
provides utility code to compute a few metrics.
"""

# Importing transformers or openai before torch (which is done when creating the model)
# can lead to a core dump.
# So import it first to avoid this issue.
import torch  # noqa: F401
import os
import warnings
from typing import Optional, Iterator, Sequence, Any
from pprint import pprint

from datasets import Dataset, load_from_disk

from xc_cache.data.get_qa_task import (  # noqa: E402
    get_qa_task,
    KNOWN_CONTEXT_OPTIONS,
    KNOWN_ANSWER_OPTIONS,
)
from xc_cache.data.format_qa_task import (  # noqa: E402
    format_qa_task,
    CleanupQATask,
    KNOWN_QA_TASK_FORMATS,
    KNOWN_POST_CLEANUPS,
)
from xc_cache.models import KNOWN_MODEL_TYPE, infer_model_type  # noqa: E402
from xc_cache.inference.get_interface import get_interface  # noqa: E402
from xc_cache.utils.scripting import get_local_rank_and_world_size  # noqa: E402
from xc_cache.utils.hf_datasets import no_cache, concatenate_datasets_with_infodict  # noqa: E402


def create_parser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", default="./", type=str, help="Path where to dump results")

    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to HF model repository (local or in HF hub).",
    )

    parser.add_argument(
        "--model_ckpt",
        type=str,
        help="Model checkpoint to be loaded with our own code.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=list(KNOWN_MODEL_TYPE.keys()),
        help="Specify model type. Inferred if left empty.",
    )

    parser.add_argument(
        "--cache_path",
        default="./tmp-cache/",
        type=str,
        help="Path for cache.",
    )

    parser.add_argument(
        "--dataset_name",
        default="xc_cache_training_data",
        type=str,
        help="Name of the original dataset used to build the task. Do not confuse 'name' with 'path'! If you want to use 'ServiceNow/foo', the 'name' is 'foo'!",
    )

    parser.add_argument(
        "--dataset_split",
        default="test",
        type=str,
        help="Split of the dataset to use.",
    )

    parser.add_argument(
        "--dataset_num_shards",
        default=1,
        type=int,
        help="Number of shards in which to break the dataset.",
    )

    parser.add_argument(
        "--model_output_path",
        type=str,
        help="Save intermediate result of what the model outputs.",
    )

    parser.add_argument(
        "--task_context",
        type=str,
        choices=list(KNOWN_CONTEXT_OPTIONS.keys()),
        help="Operations to specify the task's context (if not already explicit in dataset).",
    )

    parser.add_argument(
        "--task_answer",
        type=str,
        choices=list(KNOWN_ANSWER_OPTIONS.keys()),
        help="Operations to specify the task's answer (if not already explicit in dataset).",
    )

    parser.add_argument(
        "--task_format",
        type=str,
        choices=list(KNOWN_QA_TASK_FORMATS.keys()),
        help="Format to be given to the task for the model to consume.",
    )

    parser.add_argument(
        "--post_cleanup",
        type=str,
        choices=list(KNOWN_POST_CLEANUPS.keys()),
        help="Cleanup operations done after the model.",
    )

    parser.add_argument(
        "--include_title",
        action="store_true",
        default=False,
        help="Whether to include context titles in the model inputs. Only supported by FiD atm.",
    )

    parser.add_argument(
        "--exclude_context",
        action="store_true",
        default=False,
        help="Whether to exclude context in the model inputs. Only supported by FiD. If set, context title is also excluded.",
    )

    parser.add_argument(
        "--model_max_length",
        type=int,
        help="Maximal input size for the model. Attempts to infer it if left empty.",
    )

    parser.add_argument(
        "--max_new_tokens",
        default=30,
        type=int,
        help="Maximal number of new tokens to be generated.",
    )

    parser.add_argument(
        "--to_device",
        type=str,
        help="If a device is provided, will explicitly call `model.to(device)`. If `LOCAL_RANK` appears in the string, it will be substituted by the appropriate environment variable. This last feature's main use is `--to_device cuda:LOCAL_RANK`.",
    )

    parser.add_argument(
        "--ds_config",
        type=str,
        help="Path to deepspeed configuration file.",
    )

    parser.add_argument(
        "--peft_config",
        type=str,
        help="Path to the PEFT fine-tuning model configuration file.",
    )

    parser.add_argument(
        "--model_peft_ckpt",
        type=str,
        help="PEFT finetuning checkpoint to be loaded.",
    )

    parser.add_argument(
        "--aggregate",
        default=False,
        action="store_true",
        help="Aggregate all the shards as a single dataset. Just append this to all the arguments you used to generate the shards.",
    )

    parser.add_argument(
        "--subset_size",
        type=int,
        help="Number of samples to keep in a subset of the dataset. Deterministic and attempts to handle redundancies decently.",
    )

    parser.add_argument(
        "--filter",
        type=str,
        help="Specify conditions column0:value0,column1:value1,... to filter the dataset, e.g., `--filter dataset:nq`. Filter is applied before subset_size (if any).",
    )

    return parser


DEFAULT_KWARGS_BY_MODEL_TYPE = {
    "gptbigcode": {
        "task_format": "cross_uaf_qic",
        "post_cleanup": "cross",
    },
    "llama": {
        "task_format": "cross_llama_chat_qic",
        "post_cleanup": "cross",
    },
    "tulu": {
        "task_format": "cross_uaf_qic",
        "post_cleanup": "cross",
    },
    "mistral": {
        "task_format": "cross_instruct_qic",
        "post_cleanup": "cross",
    },
    "fid": {
        "task_format": "fid",
        "post_cleanup": "fid",
    },
}


def infer_default_kwargs(kwargs: dict[str, Any]) -> None:
    """Infer different kwarg defaults in a model_type-aware manner."""
    if not kwargs["model_type"]:
        kwargs["model_type"] = infer_model_type(
            model_path=kwargs["model_path"], model_ckpt=kwargs["model_ckpt"]
        )
    defaults = DEFAULT_KWARGS_BY_MODEL_TYPE.get(kwargs["model_type"], {})
    for key, value in defaults.items():
        if kwargs[key] is None:
            kwargs[key] = value


def get_output_ds_name(*, dataset_name, dataset_split, subset_size=None, filter=None, **kwargs):
    out = f"{dataset_name}-{dataset_split}"
    if subset_size is not None:
        out = out + f"-sub{subset_size}"
    if filter is not None:
        out = out + f"-filter_{filter}"
    return out


def get_all_shard_paths(
    *,
    output_ds_name,
    dataset_num_shards,
    model_output_path,
    **kwargs,
) -> list[str]:
    num_digits = len(str(dataset_num_shards))
    return [
        os.path.join(
            model_output_path,
            f"{output_ds_name}-{str(i).zfill(num_digits)}of{str(dataset_num_shards)}",
        )
        for i in range(dataset_num_shards)
    ]


def get_shard_iter(
    shard_paths: Sequence[str],
    formatted_task: Dataset,
    *,
    dataset_num_shards,
    local_rank,
    world_size,
    **kwargs,
) -> Iterator[tuple[str, Dataset]]:
    with no_cache():
        for i, shard_path in enumerate(shard_paths):
            if (i % world_size) == local_rank:
                shard = formatted_task.shard(
                    num_shards=dataset_num_shards, index=i, contiguous=True
                )
                yield shard_path, shard


def main(explicit_arguments: Optional[list[str]] = None) -> str:
    parser = create_parser()
    args = parser.parse_args(explicit_arguments)

    # For multiprocess. No need to set them, torchrun (or deepspeed) will handle them.
    #     world_size is the total number of workers.
    #     local_rank is the number of the present worker (in range(world_size)).
    args.local_rank, args.world_size = get_local_rank_and_world_size()

    if args.include_title and args.exclude_context:
        warnings.warn("exclude_context is True: title won't be included in the model input.")

    kwargs = vars(args)
    infer_default_kwargs(kwargs)
    pprint(kwargs)

    output_ds_name = get_output_ds_name(**kwargs)
    shard_paths = get_all_shard_paths(output_ds_name=output_ds_name, **kwargs)

    if kwargs["aggregate"]:
        # The actual work is already done, we only have to aggregate!
        # Only worker rank 0 should do this
        assert kwargs["local_rank"] == 0
        all_shards = [load_from_disk(shard_path) for shard_path in shard_paths]
        aggregated = concatenate_datasets_with_infodict(all_shards)

        aggregated.save_to_disk(os.path.join(kwargs["model_output_path"], output_ds_name))
    else:
        model_interface = get_interface(**kwargs)

        try:
            # return_context_list controls whether contexts will be concatenated or chunked by the formatter.
            kwargs.update({"return_context_list": model_interface.return_context_list})
        except AttributeError:
            # Only cross-attn model interfaces are expected to have the return_context_list attribute.
            pass

        # Do actual work
        qa_task = get_qa_task(**kwargs)
        tokenizer = getattr(model_interface, "tokenizer", None)
        max_length = kwargs.get("model_max_length", getattr(tokenizer, "model_max_length", None))

        formatted_task = format_qa_task(
            qa_task=qa_task,
            task_format=kwargs["task_format"],
            include_title=kwargs["include_title"],
            include_context=not kwargs["exclude_context"],
            tokenizer=tokenizer,
            max_length=max_length,
        )

        for shard_path, shard_in in get_shard_iter(
            shard_paths=shard_paths, formatted_task=formatted_task, **kwargs
        ):
            desc = shard_path.split("/")[-1]
            model_interface.process_dataset(
                shard_in,
                named_cache_path=shard_path,
                desc=desc,
                post_process=CleanupQATask(**kwargs),
            )

    return output_ds_name


if __name__ == "__main__":
    main()
