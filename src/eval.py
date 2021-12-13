import errant
import pandas as pd
import spacy

from src.utils import PROJ, errant_tokenize, load_gold_edits


def get_spacy_model(lang: str):
    model = None
    try:
        model = spacy.load(lang)
    except OSError:
        spacy.cli.download(lang)  # type: ignore
        model = spacy.load(lang)
    return model


annotator = errant.load("en", nlp=get_spacy_model("en"))


def _get_edits(o: str, c: str):
    orig = annotator.parse(errant_tokenize(o))
    cor = annotator.parse(errant_tokenize(c))
    alignment = annotator.align(orig, cor)
    edits = []
    for e in annotator.merge(alignment):
        e = annotator.classify(e)
        edits.append("|||".join(e.to_m2()[2:].split("|||")[:3]))
    return edits


def get_precision_recall_f05_score(gold_edits, original_sents, target_sents, predicted_sents):
    tps = []
    fps = []
    fns = []
    assert len(gold_edits) == len(predicted_sents) == len(original_sents)
    for i, gld_edits in enumerate(gold_edits):
        sent_org = original_sents[i]
        sent_hyp = predicted_sents[i]
        hyp_edits = _get_edits(sent_org, sent_hyp)

        gld_edits = set(gld_edits)
        hyp_edits = set(hyp_edits)

        tp = len(gld_edits & hyp_edits)
        fp = len(hyp_edits - gld_edits)
        fn = len(gld_edits - hyp_edits)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)

    tp = sum(tps)
    fp = sum(fps)
    fn = sum(fns)

    p = tp / (tp + fp) if fp else 1.0
    r = tp / (tp + fn) if fn else 1.0
    f05 = (
        ((1 + 0.5 ** 2) * (p * r) / (0.5 ** 2 * p + r)) if p > 0 else 0.0
    )  # sourcery skip: inline-immediately-returned-variable

    return {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f05": f05}


def main():
    with open(PROJ / "outputs/predictions.txt") as fp:
        predicted_sents = [h.strip() for h in fp.readlines()]
    eval_df = pd.read_csv(PROJ / "data/test.csv")
    original_sents = eval_df["input_text"].tolist()
    target_sents = eval_df["target_text"].tolist()
    gold_edits = load_gold_edits(PROJ / "outputs/edits-gold.txt")
    print(
        get_precision_recall_f05_score(
            gold_edits=gold_edits,
            original_sents=original_sents,
            target_sents=target_sents,
            predicted_sents=predicted_sents,
        )
    )


if __name__ == "__main__":
    main()
