import json
import os
from dataclasses import asdict
from pathlib import Path

import wandb
from dotenv import load_dotenv

load_dotenv()
PROJ_HOME = Path(__file__).parent.parent
HF_API_KEY = os.environ.get("HF_API_TOKEN")

_ERRANT_TOKENIZER_MAPPINGs = [
    (" .", "."),
    (" ,", ","),
    (" ?", "?"),
    (" !", "!"),
    (" :", ":"),
    (" ;", ";"),
    (" n't", "n't"),  # do n't
    (" 're ", "'re "),  # we 're
    (" 'm ", "'m "),  # I 'm
    (" 've ", "'ve "),  # I 've
    (" 'll ", "'ll "),  # I 'll
    (" 's ", "'s "),  # Laura 's (singular possive)
    ("s ' ", "s' "),  # years ' (plural possessive)
    (" `` ", ' "'),  # `` (left quote)
    (" '' ", '" '),  # '' (right quote)
]


def errant_detokenize(text: str) -> str:
    for orig, replacement in _ERRANT_TOKENIZER_MAPPINGs:
        text = text.replace(orig, replacement)
    return text


def errant_tokenize(text):
    for replacement, orig in _ERRANT_TOKENIZER_MAPPINGs:
        text = text.replace(orig, replacement)
    return text


def dump_args(out_path: Path, *args):
    with open(out_path, "w") as fp:
        d = {}
        for arg in args:
            for k, v in asdict(arg).items():
                d[k] = v
                if isinstance(v, Path):
                    d[k] = str(v.resolve())
        json.dump(d, fp, indent=4)


def log_metrics(args_path: Path, results_path: Path, predictions_path: Path):
    with open(args_path, "r") as fp:
        args = json.load(fp)

    with open(results_path, "r") as fp:
        results = json.load(fp)

    wandb.init(
        project="hf-writing-assistant",
        config=args,
    )
    wandb.log(results)
    wandb.log({"predictions": [p.strip() for p in open(predictions_path, "r")]})
