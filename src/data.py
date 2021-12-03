import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, load_dataset  # type: ignore
from tqdm import tqdm

from src.utils import errant_detokenize, errant_tokenize


class DatasetWriter:
    def __init__(
        self,
        name: str,
        dataset: Dataset,
        task_prefix: str = "Grammar",
    ):
        self.name = name
        self.dataset = dataset
        self.task_prefix = task_prefix.replace(":", "").strip()

    def write_csv(self, out_dir: Path):
        with open(out_dir / f"{self.name}.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames="prefix input_text target_text".split())
            writer.writeheader()
            for row in tqdm(self.dataset):
                writer.writerow(
                    {
                        "prefix": self.task_prefix,
                        "input_text": row["_input"],
                        "target_text": row["_target"],
                    }
                )

    def write_texts(self, out_dir: Path):
        with open(out_dir / f"{self.name}-input.txt", "w") as fp_input:
            with open(out_dir / f"{self.name}-target.txt", "w") as fp_target:
                for row in tqdm(self.dataset):
                    fp_input.write(row["_original"] + "\n")
                    fp_target.write(row["_corrected"] + "\n")


class DatasetLoader(ABC):
    def __init__(
        self,
        name: str,
        dataset: Dataset,
        original_col: Optional[str] = None,
        corrected_col: Optional[str] = None,
        tokenized_original_col: Optional[str] = None,
        tokenized_corrected_col: Optional[str] = None,
    ):
        self.name = name
        self._dataset = self._clean_dataset(dataset)
        self._rename_columns_(
            original_col, corrected_col, tokenized_original_col, tokenized_corrected_col
        )
        # self._add_task_prefix_(task_prefix.replace(":", "").strip())

    def get_dataset(self) -> Dataset:
        return self._dataset

    def _add_task_prefix_(self, task_prefix) -> None:
        assert (
            False
        ), "no need to add task prefix to dataset right now, since we put it in the pandas dataframe as its own column"
        self._dataset = self._dataset.map(lambda _: {"_prefix": task_prefix})

    def _rename_columns_(
        self,
        original_col: Optional[str] = None,
        corrected_col: Optional[str] = None,
        tokenized_original_col: Optional[str] = None,
        tokenized_corrected_col: Optional[str] = None,
    ) -> None:
        if original_col and corrected_col and tokenized_original_col and tokenized_corrected_col:
            self._dataset = (
                self._dataset.rename_column(original_col, "_input")  # type: ignore
                .rename_column(corrected_col, "_target")
                .rename_column(tokenized_original_col, "_original")
                .rename_column(tokenized_corrected_col, "_corrected")
            )
        for col in self._dataset.column_names:
            if col not in ["_input", "_target", "_original", "_corrected"]:
                self._dataset = self._dataset.remove_columns(col)  # type: ignore

    @abstractmethod
    def _clean_dataset(self, dataset: Dataset) -> Dataset:
        ...


class MerlinDatasetLoader(DatasetLoader):
    def __init__(self, lang: str):
        ds: Dataset = load_dataset("aseifert/merlin", data_files={"train": f"{lang}.jsonl"})["train"]  # type: ignore
        super().__init__(
            name="merlin",
            dataset=ds,
            original_col="input",
            corrected_col="target",
            tokenized_original_col="original",
            tokenized_corrected_col="corrected",
        )

    def _clean_dataset(self, dataset: Dataset) -> Dataset:
        def clean(x):
            def _clean_text(text: str):
                return text.strip()

            return {
                "original": _clean_text(x["original"]),
                "corrected": _clean_text(x["corrected"]),
            }

        def remove_empty(x):
            return x["corrected"] != ""

        def remove_identical(x):
            return x["original"] != x["corrected"]

        def create_model_data(x):
            def apply_detokenize(x):
                return {
                    "input": errant_detokenize(x["original"]),
                    "target": errant_detokenize(x["corrected"]),
                }

            detokenized = apply_detokenize(x)
            return {
                "input": detokenized["input"],
                "target": detokenized["target"],
            }

        return (
            dataset.map(clean).filter(remove_empty).filter(remove_identical).map(create_model_data)
        )


class PieDatasetLoader(DatasetLoader):
    def __init__(self, take_n: Optional[int] = None):
        if take_n:
            ds: Dataset = load_dataset("aseifert/pie-synthetic", split="train", streaming=True)  # type: ignore
            ds_iter = iter(ds)
            samples = [next(ds_iter) for _ in range(take_n)]
            ds = Dataset.from_dict(pd.DataFrame(samples).to_dict(orient="list"))
        else:
            ds: Dataset = load_dataset("aseifert/pie-synthetic", split="train", streaming=False)  # type: ignore
            # ds = ds.select(range(10_000_000))

        assert ds is not None

        super().__init__(
            name="pie",
            dataset=ds,
            original_col="input",
            corrected_col="target",
            tokenized_original_col="original",
            tokenized_corrected_col="corrected",
        )

    def _clean_dataset(self, dataset: Dataset) -> Dataset:
        def create_model_data(x):
            def apply_detokenize(x):
                return {
                    "input": errant_detokenize(x["original"]),
                    "target": errant_detokenize(x["corrected"]),
                }

            detokenized = apply_detokenize(x)
            return {
                "input": detokenized["input"],
                "target": detokenized["target"],
            }

        return dataset.map(create_model_data)


class JFLEGDatasetLoader(DatasetLoader):
    def __init__(self, split: str):
        super().__init__(
            name="jfleg",
            dataset=load_dataset("jfleg")[split],  # type: ignore
            original_col="input",
            corrected_col="target",
            tokenized_original_col="sentence",
            tokenized_corrected_col="correction",
        )

    def _clean_dataset(self, dataset: Dataset) -> Dataset:
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
                    "input": errant_detokenize(x["sentence"]),
                    "target": errant_detokenize(x["correction"]),
                }

            detokenized = apply_detokenize(x)
            return {
                "input": detokenized["input"],
                "target": detokenized["target"],
            }

        # "corrections" contains a list -- after exploding every item has its own row
        dataset = Dataset.from_pandas(dataset.to_pandas().explode("corrections", ignore_index=True))  # type: ignore
        dataset = dataset.rename_column(
            original_column_name="corrections", new_column_name="correction"  # type: ignore
        )

        return (
            dataset.map(clean).filter(remove_empty).filter(remove_identical).map(create_model_data)
        )


class _WiLocnessDatasetLoader(DatasetLoader):
    def __init__(
        self,
        dataset: Dataset,
        name: str = "wi",
    ):
        super().__init__(
            name=name,
            dataset=dataset,
            original_col="text",
            corrected_col="corrected",
            tokenized_original_col="text_tokenized",
            tokenized_corrected_col="corrected_tokenized",
        )

    def _clean_dataset(self, dataset):
        def apply_edits(x):
            text = x["text"]
            start, end, edits = x["edits"].values()
            if not start:
                return {"corrected": x["text"]}

            running = ""
            last_end = 0
            for s, e, t in zip(start, end, edits):
                running += text[last_end:s]
                running += "" if t is None else t  # TODO: why can t be None?
                last_end = e
            running += text[last_end:]
            running = running.replace("  ", " ")
            return {"corrected": running}

        def clean(x):
            return {
                "text": x["text"].replace("\n", " ").replace("  ", " ").strip(),
                "corrected": x["corrected"].replace("\n", " ").replace("  ", " ").strip(),
            }

        def tokenize(x):
            return {
                "text_tokenized": errant_tokenize(x["text"]),
                "corrected_tokenized": errant_tokenize(x["corrected"]),
            }

        return dataset.map(apply_edits).remove_columns(["edits"]).map(clean).map(tokenize)


class WiDatasetLoader(_WiLocnessDatasetLoader):
    def __init__(self, split: str):
        dd: DatasetDict = load_dataset("wi_locness", "wi")  # type: ignore
        super().__init__(name="wi", dataset=dd[split])


class LocnessDatasetLoader(_WiLocnessDatasetLoader):
    def __init__(self, split: str):
        dd: DatasetDict = load_dataset("wi_locness", "locness")  # type: ignore
        super().__init__(name="locness", dataset=dd[split])
