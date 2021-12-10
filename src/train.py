import os
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd
from simpletransformers.t5 import T5Args, T5Model  # type: ignore
from transformers import HfArgumentParser  # type: ignore

from src.eval import get_precision_recall_f05_score
from src.utils import PROJ, clean_task_prefix, load_gold_edits, set_rlimit

"""
Increase number of file descriptors that can be opened.
This is needed for torch.multiprocessing's file_descriptor based sharing strategy.
docs: https://pytorch.org/docs/master/multiprocessing.html#file-descriptor-file-descriptor
gh issue: https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
"""
set_rlimit(limit=4096)


@dataclass
class ModelArgs:
    model_name: str
    model_type: Optional[str] = None


@dataclass
class DataArgs:
    models_dir: Path = Path(os.environ.get("SM_MODEL_DIR", PROJ / "models/"))
    train_csv: Path = Path(os.environ.get("SM_CHANNEL_TRAIN_CSV", PROJ / "data/train.csv"))
    eval_csv: Optional[Path] = Path(os.environ.get("SM_CHANNEL_EVAL_CSV", PROJ / "data/eval.csv"))
    edits_gold: Optional[Path] = Path(
        os.environ.get("SM_CHANNEL_EDITS_GOLD", PROJ / "data/edits-gold.txt")
    )
    task_prefix: str = "Grammar"


@dataclass
class TrainingArgs:
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    use_cuda: bool = True
    use_wandb: bool = True


def main(model_args: ModelArgs, data_args: DataArgs, train_args: TrainingArgs) -> None:
    for path in asdict(data_args).values():
        if path and isinstance(path, Path):
            assert path.exists(), f"{path} does not exist"

    t5_args = T5Args()
    t5_args.max_length = 512
    t5_args.train_batch_size = train_args.batch_size
    t5_args.learning_rate = train_args.learning_rate
    t5_args.num_train_epochs = train_args.num_train_epochs
    if train_args.use_wandb:
        t5_args.wandb_project = "hf-writing-assistant"
    t5_args.output_dir = str(data_args.models_dir)
    t5_args.best_model_dir = str(data_args.models_dir / "best_model")
    t5_args.overwrite_output_dir = True
    t5_args.evaluate_generated_text = True
    t5_args.evaluate_during_training = True
    t5_args.evaluate_during_training_verbose = True
    # t5_args.evaluate_during_training_steps = 1000
    t5_args.use_multiprocessing = False
    default_model_type = "mt5" if "mt5" in model_args.model_name else "t5"
    model_type = (model_args.model_type or default_model_type).lower()
    model = T5Model(
        model_type=model_type,
        model_name=model_args.model_name,
        args=t5_args,
        use_cuda=train_args.use_cuda,
    )

    train_df = pd.read_csv(data_args.train_csv)
    eval_df = pd.read_csv(data_args.eval_csv) if data_args.eval_csv else None
    for df in [train_df, eval_df]:
        df["prefix"] = clean_task_prefix(data_args.task_prefix)
    original_sents = eval_df["input_text"].tolist() if eval_df is not None else None
    gold_edits = load_gold_edits(data_args.edits_gold) if data_args.eval_csv else None
    assert len(gold_edits) == len(original_sents) == len(eval_df)  # type: ignore

    def _get_precision_recall_f05_score(targets, predictions, key: str):
        with open(PROJ / "outputs/tgts.txt", "w") as fp:
            fp.write("\n".join(targets))
        with open(PROJ / "outputs/preds.txt", "w") as fp:
            fp.write("\n".join(predictions))
        return get_precision_recall_f05_score(
            gold_edits=gold_edits,
            original_sents=original_sents,
            target_sents=targets,
            predicted_sents=predictions,
        )[key]

    # precision = partial(_get_precision_recall_f05_score, key="p")
    # recall = partial(_get_precision_recall_f05_score, key="r")
    f05 = partial(_get_precision_recall_f05_score, key="f05")

    model.train_model(
        train_data=train_df,
        eval_data=eval_df,
        # p=precision,
        # r=recall,
        f05=f05,
    )
    # model.save_model(str(data_args.out))


if __name__ == "__main__":
    (model_args, data_args, train_args) = HfArgumentParser(
        [ModelArgs, DataArgs, TrainingArgs]
    ).parse_args_into_dataclasses()
    main(model_args, data_args, train_args)
