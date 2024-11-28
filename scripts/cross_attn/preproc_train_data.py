import os
import argparse
import multiprocess as mp
import datasets
from datasets import Dataset
from typing import Optional

from xc_cache.data.utils.encoder import Encoder
from xc_cache.data.utils.pre_processors import TrainDataPreProcessor


def train_data_prep(
    training_data_path: str,
    encoding_model_path: str,
    context_length: int,
    data_dir: str,
    processing_batch_size: Optional[int] = 1,
    num_proc: int = 1,
) -> Dataset:
    """Get and pre-process Training data dataset.

    Args:
        training_data_path (str): Path to dataset on hf's hub.
        encoding_model_path (str): Path to encoding model.
        context_length (int): Maximum length of ids sequence.
        data_dir (str): Path to cache and save processed data. Passing None disables caching/saving.
        processing_batch_size (Optional[int]): Batch size for pre-processing data if not previously cached. Defaults to 1.
        num_proc (int): Number of preprocessing workers. Should match the number of available GPUS.

    Returns:
        Dataset: Processed dataset.
    """

    model_base_name = os.path.basename(encoding_model_path)
    processed_dataset_save_path = os.path.join(data_dir, "processed_data", model_base_name)

    encoding_model = Encoder(
        model_name=encoding_model_path, maximum_length=context_length, num_proc=num_proc
    )

    # We expect a dataset in hf's hub containing the fields 'context', 'question', and 'answer'.
    # The training dataset being used currently can be prepared using xc_cache.data.cross_attn.prepare_train_data.py.
    training_data = datasets.load_dataset(training_data_path, cache_dir=data_dir)

    processed_datasets_dict = {}

    # We expect at least the following set of splits to be available:
    # {'train', 'val'}
    for split in training_data.keys():
        processed_data_split = training_data[split].map(
            TrainDataPreProcessor(encoding_model),
            remove_columns=training_data[split].column_names,
            batched=True,
            batch_size=processing_batch_size,
            num_proc=num_proc,
            with_rank=True,
        )

        processed_datasets_dict[split] = processed_data_split

    processed_data = datasets.DatasetDict(processed_datasets_dict)

    processed_data.save_to_disk(processed_dataset_save_path)

    return processed_data


def main(explicit_arguments: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_data_id",
        type=str,
        default="ServiceNow/xc_cache_training_data",
        help="Hugging face hub ID of training data.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Hugging face hub ID or local path to encoding model",
    )
    parser.add_argument(
        "--maximum_input_length",
        type=int,
        required=True,
        help="Maximum number of tokens to be fed into a model each time it's queried for embeddings.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Pre-processing batch size. Defaults to 1."
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes to run in parallel for map. Must be equal to the number of gpus on the node.",
    )
    parser.add_argument(
        "--data_cache_path",
        default=None,
        help="Path where to cache data.",
    )
    parser.add_argument(
        "--hub_push_path",
        help="Optional id to push the processed data to hugging face datasets.",
    )

    args = parser.parse_args(explicit_arguments)

    processed_dataset = train_data_prep(
        training_data_path=args.training_data_id,
        encoding_model_path=args.embedding_model,
        context_length=args.maximum_input_length,
        data_dir=args.data_cache_path,
        processing_batch_size=args.batch_size,
        num_proc=args.num_proc,
    )

    if args.hub_push_path is not None:
        processed_dataset.push_to_hub(args.hub_push_path)


####################################################################################################

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()

####################################################################################################
