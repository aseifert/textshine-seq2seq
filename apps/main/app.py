import time

import errant
import spacy
import streamlit as st
from happytransformer import HappyTextToText, TTSettings

from highlighter import show_highlights

checkpoints = [
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
def get_model(model_name):
    return HappyTextToText("T5", model_name)


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
    checkpoint = st.selectbox("Choose model", checkpoints)
    model = get_model(checkpoint)
    args = TTSettings(num_beams=5, min_length=1, max_length=1024)

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
            prefixed_input_text = "Grammar: " + input_text
            result = predict(model, args, prefixed_input_text)

            try:
                show_highlights(annotator, input_text, result)
                st.write("")
                st.success(result)
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
