from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests
from datasets import Dataset, load_dataset  # type: ignore
from tqdm import tqdm
from transformers import HfArgumentParser, pipeline  # type: ignore
from transformers.pipelines.base import KeyDataset

from src.utils import HF_API_KEY, PROJ_HOME, errant_tokenize


@dataclass
class ModelArgs:
    model_name: str
    max_length: int = 1024
    tokenize: bool = True
    device: int = 0


@dataclass
class DataArgs:
    test_csv: Path = PROJ_HOME / "data/test.csv"
    out_path: Path = PROJ_HOME / "predictions.txt"


class Tagger(ABC):
    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def tag(self, dataset):
        ...


class PipelineTagger(Tagger):
    def __init__(self, model_name: str, max_length: int = 1024, device: int = 0, *args, **kwargs):
        super().__init__(model_name, max_length, device, *args, **kwargs)
        self.max_length = max_length
        self.pipe = pipeline("text2text-generation", model_name, device=device)

    def tag(self, dataset):
        return self.pipe(dataset, max_length=self.max_length)


class HfAccelerateTagger(Tagger):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        assert HF_API_KEY is not None, "HF_API_KEY is not set"
        self._api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self._headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    def _correct(self, sent: str):
        prompt = f"""A non-native wrote: \"{sent}\"\nA teacher corrected that text to read: \""""
        payload = {"inputs": prompt}
        response = requests.post(self._api_url, headers=self._headers, json=payload)
        return response.json()[0]["generated_text"].rstrip('"')

    def tag(self, dataset):
        for text in dataset:
            yield self._correct(text)


def tag(tagger, dataset: Dataset, max_length: int, tokenize: bool) -> List[str]:
    key_dataset = KeyDataset(dataset, "input")  # type: ignore

    # we predict only on the unique dataset
    # FIXME: there must be a better way to construct the unique key dataset
    key_dataset_unique = KeyDataset(Dataset.from_dict({"input": dataset.unique("input")}), "input")  # type: ignore
    unique_results = [
        p["generated_text"] for p in tqdm(tagger.tag(key_dataset_unique, max_length=max_length))
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
    tagger = PipelineTagger(model_args.model_name, device=model_args.device)

    predictions = tag(
        tagger=tagger,
        dataset=dataset,
        max_length=model_args.max_length,
        tokenize=model_args.tokenize,
    )

    with open(data_args.out_path, "w") as fp:
        fp.write("\n".join(predictions))


if __name__ == "__main__":
    (model_args, data_args) = HfArgumentParser([ModelArgs, DataArgs]).parse_args_into_dataclasses()
    main(model_args, data_args)
