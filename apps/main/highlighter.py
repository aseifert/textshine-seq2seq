"""
This module is taken with slight modifications from:
https://github.com/PrithivirajDamodaran/Gramformer/blob/main/gramformer/gramformer.py
"""

import re

import pandas as pd
from annotated_text import annotated_text
from bs4 import BeautifulSoup


def show_highlights(annotator, input_text, corrected_sentence):
    strikeout = lambda x: "\u0336".join(x) + "\u0336"
    highlight_text = highlight(annotator, input_text, corrected_sentence)
    color_map = {"d": "#faa", "a": "#afa", "c": "#fea"}
    tokens = re.split(r"(<[dac]\s.*?<\/[dac]>)", highlight_text)
    annotations = []
    for token in tokens:
        soup = BeautifulSoup(token, "html.parser")
        tags = soup.findAll()
        if tags:
            _tag = tags[0].name
            _type = tags[0]["type"]
            _text = tags[0]["edit"]
            _color = color_map[_tag]
            if _tag == "d":
                _text = strikeout(tags[0].text)
            annotations.append((_text, _type, _color))
        else:
            annotations.append(token)
    annotated_text(*annotations)


def show_edits(annotator, input_text, corrected_sentence):
    edits = get_edits(annotator, input_text, corrected_sentence)
    df = pd.DataFrame(
        edits,
        columns=[
            "type",
            "original word",
            "original start",
            "original end",
            "correct word",
            "correct start",
            "correct end",
        ],
    )
    return df.set_index("type")


def highlight(annotator, orig, cor):
    edits = get_edits(annotator, orig, cor)
    orig_tokens = orig.split()
    ignore_indexes = []
    for edit in edits:
        edit_type = edit[0]
        edit_str_start = edit[1]
        edit_spos = edit[2]
        edit_epos = edit[3]
        edit_str_end = edit[4]
        for i in range(edit_spos + 1, edit_epos):
            ignore_indexes.append(i)
        if edit_str_start == "":
            if edit_spos >= 1:
                new_edit_str = orig_tokens[edit_spos - 1]
                # print("edit_spos >= 1", new_edit_str)
                edit_spos -= 1
            else:
                new_edit_str = orig_tokens[edit_spos + 1]
                # print("new", new_edit_str)
                edit_spos += 1

            if edit_type == "PUNCT":
                st = (
                    "<a type='"
                    + edit_type
                    + "' edit='"
                    + edit_str_end
                    + "'>"
                    + new_edit_str
                    + "</a>"
                )
            else:
                st = (
                    "<a type='"
                    + edit_type
                    + "' edit='"
                    + new_edit_str
                    + " "
                    + edit_str_end
                    + "'>"
                    + new_edit_str
                    + "</a>"
                )
        elif edit_str_end == "":
            st = "<d type='" + edit_type + "' edit=''>" + edit_str_start + "</d>"
        else:
            st = (
                "<c type='" + edit_type + "' edit='" + edit_str_end + "'>" + edit_str_start + "</c>"
            )
        orig_tokens[edit_spos] = st
    for i in sorted(ignore_indexes, reverse=True):
        del orig_tokens[i]
    return " ".join(orig_tokens)


def get_edits(annotator, orig, cor):
    orig = annotator.parse(orig)
    cor = annotator.parse(cor)
    alignment = annotator.align(orig, cor)
    edits = annotator.merge(alignment)
    if len(edits) == 0:
        return []
    edit_annotations = []
    for e in edits:
        e = annotator.classify(e)
        edit_annotations.append(
            (e.type[2:], e.o_str, e.o_start, e.o_end, e.c_str, e.c_start, e.c_end)
        )

    return edit_annotations or []
