from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(config_path="../config/", config_name="ensemble", version_base="1.2.0")
def _main(cfg: DictConfig):
    # model load
    lgbm_preds = pd.read_csv(
        Path(get_original_cwd()) / cfg.output.path / cfg.output.lightgbm
    )
    cb_preds = pd.read_csv(
        Path(get_original_cwd()) / cfg.output.path / cfg.output.catboost
    )

    submit = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.output.submit)
    preds = np.median([lgbm_preds[cfg.data.target], cb_preds[cfg.data.target]], axis=0)
    submit[cfg.data.target] = preds

    # save
    path = Path(get_original_cwd()) / cfg.output.path
    submit.to_csv(path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
