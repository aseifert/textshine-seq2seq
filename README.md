# Textshine

Textshine is a seq2seq model for grammatical error correction.


## Installation

```bash
conda env create -n textshine -f environment.yml && conda activate textshine
huggingface-cli login  # log in to huggingface hub (needed for uploading models)
dvc repro  # run the whole pipeline (as specified in `dvc.yaml`)
pip install -e .  # install this project as an editable dependency (so you can import stuff into notebooks, regardless where they live)
```
