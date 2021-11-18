from pathlib import Path

PWD = Path(".").parent

_JFLEG_TOKENIZER_MAPPINGs = [
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


def jfleg_detokenize(text: str) -> str:
    for orig, replacement in _JFLEG_TOKENIZER_MAPPINGs:
        text = text.replace(orig, replacement)
    return text


def jfleg_tokenize(text):
    for replacement, orig in _JFLEG_TOKENIZER_MAPPINGs:
        text = text.replace(orig, replacement)
    return text
