from pathlib import Path
from typing import Tuple

import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from features.engineering import (
    create_categorical_test,
    create_categorical_train,
    add_features,
)


def load_train_dataset(config: DictConfig) -> Tuple[pd.DataFrame]:
    """
    Load train dataset
    Args:
        config (DictConfig): config
    Returns:
        Tuple[pd.DataFrame]: train_x, train_y
    """
    path = Path(get_original_cwd()) / config.data.path
    train = pd.read_parquet(path / f"{config.data.train}.parquet")
    train = add_features(train)
    train = create_categorical_train(train, config)
    train_x = train.drop(columns=[*config.data.drop_features] + [config.data.target])
    train_y = train[config.data.target]
    return train_x, train_y


def load_test_dataset(config: DictConfig) -> pd.DataFrame:
    """
    Load test dataset
    Args:
        config (DictConfig): config
    Returns:
        pd.DataFrame: test
    """
    path = Path(get_original_cwd()) / config.data.path
    test = pd.read_parquet(path / f"{config.data.test}.parquet")
    test = add_features(test)
    test = create_categorical_test(test, config)
    test_x = test.drop(columns=[*config.data.drop_features])
    return test_x
