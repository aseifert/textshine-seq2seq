from dataclasses import dataclass
from pathlib import Path

from transformers import HfArgumentParser  # type: ignore

from src.data import JFLEGDataset, LocnessDataset, StackedDataset, WiDataset

PWD = Path(__file__).parent
PROJ_HOME = PWD.parent


@dataclass
class DataArgs:
    out_dir: Path = PROJ_HOME / "data/"
    task_prefix: str = "Grammar"


def main(data_args: DataArgs):
    train_datasets = [
        JFLEGDataset("validation"),
        # WiDataset("train"),
        # WiDataset("validation"),
        # LocnessDataset("validation"),
    ]
    test_datasets = [
        JFLEGDataset("test"),
    ]

    stacked_train_dataset = StackedDataset("train", train_datasets)
    stacked_test_dataset = StackedDataset("test", test_datasets)  # type: ignore

    for stacked in (stacked_train_dataset, stacked_test_dataset):
        stacked.write_csv(data_args.out_dir)
        stacked.write_texts(data_args.out_dir)


if __name__ == "__main__":
    (data_args,) = HfArgumentParser(DataArgs).parse_args_into_dataclasses()
    main(data_args)
