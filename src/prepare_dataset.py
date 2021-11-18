from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset  # type: ignore
from transformers import HfArgumentParser  # type: ignore

from src.utils import jfleg_detokenize

PWD = Path(__file__).parent
PROJ_HOME = PWD.parent


@dataclass
class DataArgs:
    out_dir: Path = PROJ_HOME / "data/"
    task_prefix: str = "proofread"


def prepare_dataset(data_args: DataArgs) -> None:
    jfleg = load_dataset("jfleg")
    for split, dataset in jfleg.items():  # type: ignore
        # "corrections" contains a list -- after exploding every item has its own row
        dataset = Dataset.from_pandas(dataset.to_pandas().explode("corrections").reset_index())
        dataset.rename_column_(original_column_name="corrections", new_column_name="correction")

        dataset = _clean_dataset(dataset, task_prefix=data_args.task_prefix)

        df: pd.DataFrame = dataset.to_pandas()  # type: ignore
        df["input target".split()].to_csv(data_args.out_dir / f"{split}.csv", index=False)

        with open(data_args.out_dir / f"{split}-input.txt", "w") as fp:
            fp.write("\n".join(df["sentence"].tolist()))

        with open(data_args.out_dir / f"{split}-target.txt", "w") as fp:
            fp.write("\n".join(df["correction"].tolist()))


def _clean_dataset(dataset, task_prefix):
    task_prefix = task_prefix.replace(":", "").strip() + ": "

    def clean(x):
        def _clean_text(text: str):
            return text.strip()

        return {
            "sentence": _clean_text(x["sentence"]),
            "correction": _clean_text(x["correction"]),
        }

    def remove_empty(x):
        return x["correction"] != ""

    def remove_identical(x):
        return x["sentence"] != x["correction"]

    def create_model_data(x):
        def apply_detokenize(x):
            return {
                "input": jfleg_detokenize(x["sentence"]),
                "target": jfleg_detokenize(x["correction"]),
            }

        detokenized = apply_detokenize(x)
        return {
            "input": task_prefix + detokenized["input"],
            "target": detokenized["target"],
        }

    return dataset.map(clean).filter(remove_empty).filter(remove_identical).map(create_model_data)


if __name__ == "__main__":

    (data_args,) = HfArgumentParser(DataArgs).parse_args_into_dataclasses()
    prepare_dataset(data_args)
