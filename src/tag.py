from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from datasets import Dataset, load_dataset  # type: ignore
from tqdm import tqdm
from transformers import HfArgumentParser, pipeline  # type: ignore
from transformers.pipelines.base import KeyDataset

from src.utils import jfleg_tokenize

PWD = Path(__file__).parent
PROJ_HOME = PWD.parent


@dataclass
class ModelArgs:
    model_name: str
    max_length: int = 1024
    tokenize: bool = True


@dataclass
class DataArgs:
    test_csv: Path = PROJ_HOME / "data/test.csv"
    out: Path = PROJ_HOME / "predictions.txt"


def tag(pipe, dataset: Dataset, max_length: int, tokenize: bool) -> List[str]:
    dataset = KeyDataset(dataset, "input")  # type: ignore
    results = [p["generated_text"] for p in tqdm(pipe(dataset, max_length=max_length))]
    return results if not tokenize else [jfleg_tokenize(r) for r in results]


def main(
    model_args: ModelArgs,
    data_args: DataArgs,
) -> None:
    for path in asdict(data_args).values():
        assert not path or path.exists(), f"{path} does not exist"

    dataset: Dataset = load_dataset("csv", data_files={"test": str(data_args.test_csv)})["test"]  # type: ignore
    pipe = pipeline("text2text-generation", model_args.model_name, device=0)

    predictions = tag(
        pipe=pipe,
        dataset=dataset,
        max_length=model_args.max_length,
        tokenize=model_args.tokenize,
    )

    with open(data_args.out, "w") as fp:
        fp.write("\n".join(predictions))


if __name__ == "__main__":
    (model_args, data_args) = HfArgumentParser([ModelArgs, DataArgs]).parse_args_into_dataclasses()
    main(model_args, data_args)
