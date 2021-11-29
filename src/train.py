from dataclasses import asdict, dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from simpletransformers.t5 import T5Args, T5Model
from transformers import HfArgumentParser  # type: ignore

from src.eval import get_precision_recall_f05_score
from src.utils import PROJ, dump_args, load_gold_edits

torch.multiprocessing.set_sharing_strategy("file_system")

PWD = Path(__file__).parent
PROJ_HOME = PWD.parent


@dataclass
class ModelArgs:
    model_name: str
    model_type: Optional[str] = None


@dataclass
class DataArgs:
    train_csv: Path = PROJ_HOME / "data/train.csv"
    eval_csv: Optional[Path] = PROJ_HOME / "data/test.csv"
    out: Path = PROJ_HOME / "models/"


@dataclass
class TrainingArgs:
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    use_cuda: bool = True
    use_wandb: bool = True


def main(model_args: ModelArgs, data_args: DataArgs, train_args: TrainingArgs) -> None:
    for path in asdict(data_args).values():
        assert not path or path.exists(), f"{path} does not exist"
    # assert data_args.out
    # dump_args(data_args.out / "args.json", model_args, data_args, train_args)

    t5_model_args = T5Args()
    t5_model_args.min
    t5_model_args.train_batch_size = train_args.batch_size
    t5_model_args.learning_rate = train_args.learning_rate
    t5_model_args.num_train_epochs = train_args.num_train_epochs
    if train_args.use_wandb:
        t5_model_args.wandb_project = "hf-writing-assistant"
    t5_model_args.output_dir = str(data_args.out)
    t5_model_args.overwrite_output_dir = True
    t5_model_args.evaluate_generated_text = True
    t5_model_args.evaluate_during_training = True
    t5_model_args.evaluate_during_training_verbose = True
    # t5_model_args.evaluate_during_training_steps = 1000
    default_model_type = "mt5" if "mt5" in model_args.model_name else "t5"
    model_type = (model_args.model_type or default_model_type).lower()
    model = T5Model(
        model_type=model_type,
        model_name=model_args.model_name,
        args=t5_model_args,
        use_cuda=train_args.use_cuda,
    )

    train_df = pd.read_csv(data_args.train_csv)
    eval_df = pd.read_csv(data_args.eval_csv) if data_args.eval_csv else None
    original_sents = eval_df["input_text"].tolist() if data_args.eval_csv else None
    gold_edits = (
        load_gold_edits(PROJ / "outputs/edits-gold.txt") if data_args.eval_csv else None
    )
    assert len(gold_edits) == len(original_sents) == len(eval_df)

    @lru_cache
    def _get_precision_recall_f05_score(targets, predictions, key: str):
        return get_precision_recall_f05_score(
            gold_edits=gold_edits,
            original_sents=original_sents,
            target_sents=targets,
            predicted_sents=predictions,
        )[key]

    precision = partial(_get_precision_recall_f05_score, key="p")
    recall = partial(_get_precision_recall_f05_score, key="r")
    f05 = partial(_get_precision_recall_f05_score, key="f05")

    model.train_model(
        train_data=train_df,
        eval_data=eval_df,
        p=precision,
        r=recall,
        f05=f05,
    )
    # model.save_model(str(data_args.out))


if __name__ == "__main__":
    (model_args, data_args, train_args) = HfArgumentParser(
        [ModelArgs, DataArgs, TrainingArgs]
    ).parse_args_into_dataclasses()
    main(model_args, data_args, train_args)
