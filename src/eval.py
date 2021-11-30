import errant

annotator = errant.load("en")


def _get_edits(o: str, c: str):
    orig = annotator.parse(errant_tokenize(o))
    cor = annotator.parse(errant_tokenize(c))
    alignment = annotator.align(orig, cor)
    edits = []
    for e in annotator.merge(alignment):
        e = annotator.classify(e)
        edits.append("|||".join(e.to_m2()[2:].split("|||")[:3]))
    return edits


def get_precision_recall_f05_score(
    gold_edits, original_sents, target_sents, predicted_sents
):
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

    return {"p": p, "r": r, "f05": f05}
