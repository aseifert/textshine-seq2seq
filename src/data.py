from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd
from datasets import Dataset, interleave_datasets, load_dataset  # type: ignore

from src.utils import errant_detokenize, errant_tokenize


class ABCDataset(ABC):
    def __init__(
        self,
        name: str,
        dataset: Dataset,
        original_col: Optional[str] = None,
        corrected_col: Optional[str] = None,
        tokenized_original_col: Optional[str] = None,
        tokenized_corrected_col: Optional[str] = None,
        task_prefix: str = "Grammar:",
    ):
        self.name = name
        self.dataset = self._clean_dataset(dataset)
        self._rename_columns_(
            original_col, corrected_col, tokenized_original_col, tokenized_corrected_col
        )
        self._add_task_prefix_(task_prefix.replace(":", "").strip() + ": ")

    def _add_task_prefix_(self, task_prefix) -> None:
        self.dataset = self.dataset.map(lambda _: {"_prefix": task_prefix})

    def _rename_columns_(
        self,
        original_col: Optional[str] = None,
        corrected_col: Optional[str] = None,
        tokenized_original_col: Optional[str] = None,
        tokenized_corrected_col: Optional[str] = None,
    ) -> None:
        if (
            original_col
            and corrected_col
            and tokenized_original_col
            and tokenized_corrected_col
        ):
            self.dataset.rename_column_(original_col, "_input")
            self.dataset.rename_column_(corrected_col, "_target")
            self.dataset.rename_column_(tokenized_original_col, "_original")
            self.dataset.rename_column_(tokenized_corrected_col, "_corrected")

        # we need to go the pandas route because we want to keep the column order
        keep_cols = ["_input", "_target", "_original", "_corrected"]
        df: pd.DataFrame = self.dataset.to_pandas()[keep_cols]  # type: ignore
        # df.columns = [c.lstrip("_") for c in keep_cols]
        self.dataset = Dataset.from_pandas(df)

    def get_original(self) -> List[str]:
        return self.dataset["_original"]  # type: ignore

    def get_corrected(self) -> List[str]:
        return self.dataset["_corrected"]  # type: ignore

    def get_two_column_df(self) -> pd.DataFrame:
        return self.dataset.to_pandas()[["_input", "_target"]]  # type: ignore

    def write_csv(self, out_dir: Path):
        df = self.get_two_column_df()
        df["prefix"] = "Grammar: "
        df["prefix _input _target".split()].to_csv(
            out_dir / f"{self.name}.csv",
            index=False,
            header=["prefix", "input_text", "target_text"],
        )

    def write_texts(self, out_dir: Path):
        with open(out_dir / f"{self.name}-input.txt", "w") as fp:
            fp.write("\n".join(self.get_original()))

        with open(out_dir / f"{self.name}-target.txt", "w") as fp:
            fp.write("\n".join(self.get_corrected()))

    @abstractmethod
    def _clean_dataset(self, dataset: Dataset) -> Dataset:
        ...


class StackedDataset(ABCDataset):
    def __init__(self, name: str, datasets: List[ABCDataset]):
        self._datasets = datasets
        super().__init__(
            name=name, dataset=interleave_datasets([d.dataset for d in datasets])
        )

    def _clean_dataset(self, dataset):
        return dataset  # do nothing

    def _add_task_prefix_(self, task_prefix):
        # do nothing
        pass


class JFLEGDataset(ABCDataset):
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
        dataset.rename_column_(
            original_column_name="corrections", new_column_name="correction"
        )

        return (
            dataset.map(clean)
            .filter(remove_empty)
            .filter(remove_identical)
            .map(create_model_data)
        )


class _WiLocnessDataset(ABCDataset):
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
                "corrected": x["corrected"]
                .replace("\n", " ")
                .replace("  ", " ")
                .strip(),
            }

        def tokenize(x):
            return {
                "text_tokenized": errant_tokenize(x["text"]),
                "corrected_tokenized": errant_tokenize(x["corrected"]),
            }

        return (
            dataset.map(apply_edits).remove_columns(["edits"]).map(clean).map(tokenize)
        )


class WiDataset(_WiLocnessDataset):
    def __init__(self, split: str):
        dd: DatasetDict = load_dataset("wi_locness", "wi")  # type: ignore
        super().__init__(name="wi", dataset=dd[split])


class LocnessDataset(_WiLocnessDataset):
    def __init__(self, split: str):
        dd: DatasetDict = load_dataset("wi_locness", "locness")  # type: ignore
        super().__init__(name="locness", dataset=dd[split])
