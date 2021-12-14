set dotenv-load := true

default:
    just --list

train-sagemaker:
    python -m src.train_sagemaker

upload-to-hub hub_repo_name:
    #!/usr/bin/env python3
    from simpletransformers import T5
    model = T5(model_name="models/")
    model.model.push_to_hub("{{hub_repo_name}}", use_auth_token=True)
    model.tokenizer.push_to_hub("{{hub_repo_name}}", use_auth_token=True)
