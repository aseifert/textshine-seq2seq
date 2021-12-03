import time

import errant
import spacy
import streamlit as st
from fastT5 import export_and_get_onnx_model, get_onnx_model
from src.utils import errant_tokenize
from transformers import AutoTokenizer  # type: ignore

from highlighter import show_highlights

model_names = [
    "aseifert/t5-base-jfleg-wi",
    "aseifert/byt5-base-jfleg-wi",
    "team-writing-assistant/t5-base-c4jfleg",
    "Modfiededition/t5-base-fine-tuned-on-jfleg",
    "prithivida/grammar_error_correcter_v2",
]


@st.cache
def download_spacy_model(model="en"):
    try:
        spacy.load(model)
    except OSError:
        spacy.cli.download(model)  # type: ignore
    return True


@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer(model_name: str):
    model = None
    try:
        model = get_onnx_model(model_name)
    except AssertionError as e:
        model = export_and_get_onnx_model(model_name)
    assert model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


@st.cache(allow_output_mutation=True)
def get_annotator(lang: str):
    return errant.load(lang)


def predict(model, args, text: str):
    return model.generate_text(text, args=args).text


def main():
    st.title("🤗 Writing Assistant")
    st.markdown(
        """This writing assistant will proofread any text for you! See my [GitHub repo](https://github.com/aseifert/hf-writing-assistant) for implementation details."""
    )

    download_spacy_model()
    annotator = get_annotator("en")
    model_name = st.selectbox("Choose model", model_names)
    model, tokenizer = load_model_and_tokenizer(model_name)

    default_text = "A dog is bigger then mouse."
    default_text = "it gives him many apprtunites in the life, and i think that being knowledge person is a very wouderful thing to have so we can spend our lives in a successful way and full of happenis."
    input_text = st.text_area(
        label="Original text",
        value=default_text,
    )

    start = None
    if st.button("✍️ Check"):
        start = time.time()
        with st.spinner("Checking for errors 🔍"):
            output_text = predict(model, tokenizer, input_text)

            try:
                tokenized_input_text = errant_tokenize(input_text)
                tokenized_output_text = errant_tokenize(output_text)
                show_highlights(annotator, tokenized_input_text, tokenized_output_text)
                st.write("")
                st.success(output_text)
            except Exception as e:
                st.error("Some error occured!" + str(e))
                st.stop()

    st.write("---")
    st.markdown(
        "Built by [@therealaseifert](https://twitter.com/therealaseifert) during the HF community event – 👨\u200d💻 [GitHub repo](https://github.com/aseifert/hf-writing-assistant) – 🤗 Team Writing Assistant"
    )
    st.markdown(
        "_Highlighting code thanks to [Gramformer](https://github.com/PrithivirajDamodaran/Gramformer)_"
    )

    if start is not None:
        st.text(f"prediction took {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
