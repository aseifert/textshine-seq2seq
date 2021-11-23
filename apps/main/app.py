import time

import errant
import spacy
import streamlit as st
from happytransformer import HappyTextToText, TTSettings

from highlighter import show_edits, show_highlights

checkpoints = [
    "aseifert/t5-base-jfleg-wi",
    "aseifert/byt5-base-jfleg-wi",
    "prithivida/grammar_error_correcter_v2",
    "Modfiededition/t5-base-fine-tuned-on-jfleg",
]


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_model(model_name):
    return HappyTextToText("T5", model_name)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_annotator(lang: str):
    return errant.load(lang)


def output(model, args, annotator, input_text):
    with st.spinner("Checking for errors 🔍"):
        prefixed_input_text = "Grammar: " + input_text
        result = model.generate_text(prefixed_input_text, args=args).text

        try:
            st.success(result)
            show_highlights(annotator, input_text, result)
            # st.table(show_edits(annotator, input_text, result))
        except Exception as e:
            st.error("Some error occured!" + str(e))
            st.stop()


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
        output(model, args, annotator, input_text)

    st.write("---")
    st.markdown(
        f"Built by [@aseifert](https://twitter.com/therealaseifert) during the HF community event – [GitHub repo](https://github.com/aseifert/hf-writing-assistant) – Team Writing Assistant"
    )
    if start is not None:
        st.text(f"prediction took {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
