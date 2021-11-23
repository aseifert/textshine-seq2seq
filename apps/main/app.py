import time

import errant
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


st.title("🤗 Writing Assistant")
st.markdown(
    """This writing assistant will proofread any text for you! See my [GitHub repo](https://github.com/aseifert/hf-writing-assistant) for implementation details."""
)

checkpoint = st.selectbox("Choose model", checkpoints)
happy_tt = get_model(checkpoint)
annotator = get_annotator("en")
args = TTSettings(num_beams=5, min_length=1, max_length=1024)

default_text = "A dog is bigger then mouse."
default_text = "A dog is bigger then mouse."
input_text = st.text_area(
    label="Original text",
    value=default_text,
    placeholder="Enter your text here",
)
button = st.button("✍️ Check")


def output(input_text):
    with st.spinner("Checking for errors 🔍"):
        prefixed_input_text = "Grammar: " + input_text
        result = happy_tt.generate_text(prefixed_input_text, args=args).text

        try:
            st.success(result)
            show_highlights(annotator, input_text, result)
            # st.table(show_edits(annotator, input_text, result))
        except Exception as e:
            st.error("Some error occured!" + str(e))
            st.stop()


start = time.time()
output(input_text)
st.write("---")
st.text(f"Built by Team Writing Assistant ❤️ – prediction took {time.time() - start:.2f}s")
