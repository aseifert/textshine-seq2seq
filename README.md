# HF Writing Assistant

This repo contains my contribution to the HuggingFace community project.

Some steps you might want to take:

```bash
export HF_ENV_NAME="hf_writing_assistant"
conda env create -n $HF_ENV_NAME -f environment.yml  # create environment
conda activate $HF_ENV_NAME
dvc repro  # run the whole pipeline (as described in `dvc.yaml`)
pip install -e .  # install this project as an editable dependency (so you can import stuff into notebooks, regardless where they live)
```
