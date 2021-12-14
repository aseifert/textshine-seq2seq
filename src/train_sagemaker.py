from sagemaker.huggingface import HuggingFace

from src.utils import PROJ
from src.utils_sagemaker import WANDB_API_KEY, init_sagemaker, prepare_dir

if __name__ == "__main__":
    SAGEMAKER_PATH = PROJ / "sm"
    LOCAL_SRC_PATH = PROJ / "src"
    SAGEMAKER_SRC_PATH = SAGEMAKER_PATH / "src"
    prepare_dir(
        sagemaker_path=SAGEMAKER_PATH,
        local_src_path=LOCAL_SRC_PATH,
        sagemaker_src_path=SAGEMAKER_SRC_PATH,
    )

    role, _ = init_sagemaker("AlexSagemaker")

    INSTANCES = {
        "gpu": "ml.p3.2xlarge",
        "gpu-huge": "ml.g4dn.16xlarge",
        "gpu-fast-launch": "ml.g4dn.xlarge",
    }
    DEFAULT_INSTANCE = INSTANCES["gpu-fast-launch"]

    hyperparameters = {
        "batch_size": 16,
        "model_name": "t5-base",
        "learning_rate": 5e-4,
        "num_train_epochs": 10,
        "task_prefix": "Grammar",
    }

    huggingface_estimator = HuggingFace(
        source_dir=str(SAGEMAKER_PATH),
        entry_point="src/train.py",
        base_job_name="huggingface-sdk-extension",
        instance_type=INSTANCES["gpu-fast-launch"],
        instance_count=1,
        transformers_version="4.6.1",
        pytorch_version="1.7.1",
        py_version="py36",
        role=role,
        hyperparameters=hyperparameters,
        environment={"WANDB_API_KEY": WANDB_API_KEY},
    )

    huggingface_estimator.fit(
        {
            "data": "s3://alex.apollo.ai/smart/",
        },
    )
