# HF Writing Assistant

This repo contains my contribution to the HuggingFace community project.

Some steps you might want to take:

```bash
ENV_NAME="hf_writing_assistant"
conda env create -n $ENV_NAME -f environment.yml  # create environment
conda activate $ENV_NAME
dvc repro  # run the whole pipeline (as described in `dvc.yaml`)
```
