import difflib
import time

import streamlit as st
from annotated_text import annotated_text
from happytransformer import HappyTextToText, TTSettings

checkpoints = ["aseifert/t5-base-jfleg-wi", "aseifert/byt5-base-jfleg-wi"]


def diff_strings(a, b):
    result = []
    diff = difflib.Differ().compare(a.split(), b.split())
    replacement = ""
    for line in diff:
        if line.startswith("  "):
            result.append(" ")
            if len(replacement) != 0:
                result.append(("", replacement, "#ffd"))
                replacement = ""
            result.append(line[2:])
        elif line.startswith("- "):
            if len(replacement) == 0:
                replacement = line[2:]
            else:
                result.append(" ")
                result.append(("", replacement, "#fdd"))
                replacement = ""
        elif line.startswith("+ "):
            if len(replacement) == 0:
                result.append(("", line[2:], "#dfd"))
            else:
                result.append(" ")
                result.append((line[2:], "", "#ddf"))
                replacement = ""
    return result


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_model(model_name):
    # st.info(f"Loading the HappyTextToText model {model_name}, please wait...")
    return HappyTextToText("T5", model_name)


st.title("Check & Improve English Grammar")
st.markdown("This writing assistant detects 🔍 and corrects ✍️ grammatical mistakes for you!")

checkpoint = st.selectbox("Choose model", checkpoints)
happy_tt = get_model(checkpoint)
args = TTSettings(num_beams=5, min_length=1, max_length=1024)

input_text = st.text_area(
    label="Original text",
    value="Speed of light is fastest then speed of sound",
    placeholder="Enter your text here",
)
button = st.button("✍️ Check")


def output(input_text):
    with st.spinner("Checking for errors 🔍"):
        input_text = "Grammar: " + input_text
        start = time.time()
        result = happy_tt.generate_text(input_text, args=args)
        diff = diff_strings(input_text[9:], result.text)
        annotated_text(*diff)
        # st.success(result.text)
        st.write("")
        st.info(f"Correction took {time.time() - start:.2f}s")


st.markdown("**Corrected text**")
output(input_text)

st.text("Built by Team Writing Assistant ❤️")
