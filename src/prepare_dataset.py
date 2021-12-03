from dataclasses import dataclass
from pathlib import Path

from datasets import concatenate_datasets  # type: ignore
from transformers import HfArgumentParser  # type: ignore

from src.data import (
    DatasetWriter,
    JFLEGDatasetLoader,
    LocnessDatasetLoader,
    MerlinDatasetLoader,
    PieDatasetLoader,
    WiDatasetLoader,
)

PWD = Path(__file__).parent
PROJ_HOME = PWD.parent


@dataclass
class DataArgs:
    out_dir: Path = PROJ_HOME / "data/"
    task_prefix: str = "Grammar"


def main(data_args: DataArgs):
    jfleg = JFLEGDatasetLoader("validation").get_dataset().shuffle(42)
    jfleg_eval = jfleg.select(range(1000))
    jfleg_train = jfleg.select(range(1001, len(jfleg)))
    datasets = {
        "train": concatenate_datasets(
            [
                jfleg_train,
                PieDatasetLoader(take_n=50_000).get_dataset(),
                # MerlinDatasetLoader("german").get_dataset(),
                # WiDatasetLoader("train").get_dataset(),
                # WiDatasetLoader("validation").get_dataset(),
                # LocnessDatasetLoader("validation").get_dataset(),
            ],
        ),
        "eval": concatenate_datasets([jfleg_eval]),
        "test": concatenate_datasets(
            [
                JFLEGDatasetLoader("test").get_dataset(),
            ]
        ),
    }

    for name, stacked in datasets.items():
        print(name, stacked)
        writer = DatasetWriter(name, stacked, data_args.task_prefix)
        write_texts = name != "train"
        writer.write_files(data_args.out_dir, write_csv=True, write_texts=write_texts)


if __name__ == "__main__":
    (data_args,) = HfArgumentParser(DataArgs).parse_args_into_dataclasses()
    main(data_args)
