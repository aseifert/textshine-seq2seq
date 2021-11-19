import json
from dataclasses import asdict
from pathlib import Path

import wandb

PWD = Path(".").parent

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
    (" `` ", ' "'),
    (" '' ", '" '),
    # (" v", "n't"),
    # ("2 0 0 6", "2006"),
    # ("5 5", "55"),
    # ("4 0 0", "400"),
    # ("1 7-5 0", "1750"),
    # ("2 0 %", "20%"),
    # ("5 0", "50"),
    # ("1 2", "12"),
    # ("1 0", "10"),
    # ('" ballast water', '"ballast water'),
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
