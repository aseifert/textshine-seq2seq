import random
import sys
from pathlib import Path

import pandas as pd

PWD = Path(__file__).parent.resolve()
PROJ = PWD.parent.resolve()

sys.path.insert(0, str(PROJ))
from src.data import JFLEGDatasetLoader

test_df = pd.read_csv(PWD / "data/jfleg_test_head.csv")
ds = JFLEGDatasetLoader("test").get_dataset()


def test_sample():
    random_indices = random.sample(range(len(test_df)), 10)
    for i in random_indices:
        assert test_df.iloc[i]["target_text"] == ds[i]["_target"]  # type: ignore
