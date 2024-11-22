import datasets
from torch.utils.data import Dataset
from dssk.data.cross_attn.datasets_loader import DatasetWithContext


class DSSKDatasetWithContext(Dataset):
    def __init__(
        self,
        train_dataset,
        tokenizer,
        context_length,
    ):
        """Instantiates an indexed dataset wrapping a base data source and contexts."""
        self.train_dataset = train_dataset.with_format("torch")
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.include_context = True

    def __len__(self) -> int:
        return len(self.train_dataset)

    def __getitem__(self, i: int):
        d = self.train_dataset[i]

        # This code is based on research-dssk: tulu2_prompt_format()
        answer = d.get("answer", "")
        assert answer  # Both None and "" are illegal.
        # Using just a space as the separator
        combined_context = " ".join(context_text for context_text in d["contexts_list"])
        # prefix = f"<|system|>\n{combined_context}\n" if (combined_context and self.include_context) else ""
        # input_str = f"{prefix}<|user|>\n{d['question']}\n<|assistant|>\n"
        # input_str = f"{input_str}{answer}"

        # Prompt for the Llama2 model
        B_INST, E_INST = "[INST]", "\n[/INST]\n"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        input_str = (
            B_INST
            + " "
            + B_SYS
            + "You're an useful assistant.\n"
            + E_SYS
            + f"{combined_context}\nQuestion: {d['question']}\n"
            + E_INST
            + "\nAnswer: "
        )
        input_str = f"{input_str}{answer}"

        input_ids = self.tokenizer(
            input_str,
            max_length=self.context_length,
            truncation=True,
        )["input_ids"]

        return {
            "input_ids": input_ids,
        }


def dssk_data_prep(
    tokenizer,
    context_length,
    dataset_name,
    num_val_samples,
    model_type,
    dataset_subset=None,
    cross_format=False,
    data_cache_dir=None,
):
    print("DATASET: ", dataset_name)
    print("DATASET subset: ", dataset_subset)

    raw_ds = datasets.load_dataset(dataset_name, cache_dir=data_cache_dir)
    if dataset_subset:
        filtered_ds = raw_ds.filter(lambda e: e["dataset"] == dataset_subset)
    else:
        filtered_ds = raw_ds

    if cross_format:
        training_data = DatasetWithContext(
            filtered_ds["train"],
            context_length=context_length,
            tokenizer=tokenizer,
            include_context_ids=False,
            include_questions_on_contexts=False,
            chunked_contexts=False,
            return_answers=False,
            model_type=model_type,
        )
    else:
        training_data = DSSKDatasetWithContext(filtered_ds["train"], tokenizer, context_length)

    # HF's trainer will run evaluation independetly for each dataset in a dictionary
    # following the template {dataset_name: dataset}
    # If 'dataset_subset' is not None, then that dictionary will have a single dataset.

    validation_data = {}
    full_validation_data = filtered_ds["val"]
    dataset_names = set(full_validation_data["dataset"])

    for data_name in dataset_names:
        validation_data_subset = full_validation_data.filter(
            lambda example: example["dataset"] == data_name
        )

        if len(validation_data_subset) > num_val_samples:
            validation_data_subset = validation_data_subset.select(range(num_val_samples))

        if cross_format:
            validation_data[data_name] = DatasetWithContext(
                validation_data_subset,
                context_length=context_length,
                tokenizer=tokenizer,
                include_context_ids=False,
                include_questions_on_contexts=False,
                chunked_contexts=False,
                return_answers=True,
                model_type=model_type,
            )
        else:
            validation_data[data_name] = DSSKDatasetWithContext(
                validation_data_subset, tokenizer, context_length
            )

    return training_data, validation_data
