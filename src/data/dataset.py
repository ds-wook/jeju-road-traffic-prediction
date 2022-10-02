from pathlib import Path

import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


def load_train_dataset(config: DictConfig) -> pd.DataFrame:
    path = Path(get_original_cwd()) / config.data.path / config.data.train
    return pd.read_csv(path)


def load_test_dataset(config: DictConfig) -> pd.DataFrame:
    path = Path(get_original_cwd()) / config.data.path / config.data.test
    return pd.read_csv(path)
