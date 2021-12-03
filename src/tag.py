from dataclasses import dataclass
from pathlib import Path
from typing import List

from datasets import Dataset, load_dataset  # type: ignore
from tqdm import tqdm
from transformers import HfArgumentParser, pipeline  # type: ignore
from transformers.pipelines.base import KeyDataset  # type: ignore

from src.utils import clean_task_prefix, errant_tokenize

PWD = Path(__file__).parent
PROJ_HOME = PWD.parent


@dataclass
class ModelArgs:
    model_name: str
    max_length: int = 512
    tokenize: bool = True
    device: int = 0
    batch_size: int = 8


@dataclass
class DataArgs:
    test_csv: Path = PROJ_HOME / "data/test.csv"
    out_path: Path = PROJ_HOME / "predictions.txt"
    task_prefix: str = "Grammar"


def tag(pipe, dataset: Dataset, max_length: int, task_prefix: str, tokenize: bool) -> List[str]:
    dataset = dataset.map(lambda x: {"input_text": task_prefix + ": " + x["input_text"]})
    key_dataset = KeyDataset(dataset, "input_text")  # type: ignore

    # we predict only on the unique dataset
    # FIXME: there must be a better way to construct the unique key dataset
    key_dataset_unique = KeyDataset(Dataset.from_dict({"input_text": dataset.unique("input_text")}), "input_text")  # type: ignore
    unique_results = [
        p["generated_text"]
        for p in tqdm(pipe(key_dataset_unique, max_length=max_length), "tagging text")
    ]

    # and then we construct the full results out of the saved predictions
    results_map = {key_dataset_unique[i]: unique_results[i] for i in range(len(unique_results))}
    full_results = [results_map[text] for text in key_dataset]

    return full_results if not tokenize else [errant_tokenize(r) for r in full_results]


def main(
    model_args: ModelArgs,
    data_args: DataArgs,
) -> None:
    dataset: Dataset = load_dataset("csv", data_files={"test": str(data_args.test_csv)})["test"]  # type: ignore
    pipe = pipeline("text2text-generation", model_args.model_name, device=model_args.device)

    predictions = tag(
        pipe=pipe,
        dataset=dataset,
        max_length=model_args.max_length,
        task_prefix=clean_task_prefix(data_args.task_prefix),
        tokenize=model_args.tokenize,
    )

    with open(data_args.out_path, "w") as fp:
        fp.write("\n".join(predictions))


if __name__ == "__main__":
    (model_args, data_args) = HfArgumentParser([ModelArgs, DataArgs]).parse_args_into_dataclasses()
    main(model_args, data_args)
