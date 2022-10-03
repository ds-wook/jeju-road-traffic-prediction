import gc
from pathlib import Path
from typing import NoReturn

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


def csv_to_parquet(config: DictConfig) -> NoReturn:
    path = Path(get_original_cwd()) / config.data.path
    train = pd.read_csv(path / f"{config.data.train}.csv")
    test = pd.read_csv(path / f"{config.data.test}.csv")

    train.to_parquet(path / f"{config.data.train}.parquet")
    print("train done.")
    test.to_parquet(path / f"{config.data.test}.parquet")
    print("test done.")
    del train, test
    gc.collect()


@hydra.main(config_path="../config/", config_name="data", version_base="1.2.0")
def _main(cfg: DictConfig):
    csv_to_parquet(cfg)


if __name__ == "__main__":
    _main()
