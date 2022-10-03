import os
from datetime import date
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.dataset import load_test_dataset
from models.infer import inference, load_model

today = str(date.today())

if not (os.path.isdir(today)):
    os.makedirs(os.path.join(today))


@hydra.main(config_path="../config/", config_name="predict", version_base="1.2.0")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd()) / cfg.output.path

    # model load
    results = load_model(cfg, cfg.models.result)

    test_x = load_test_dataset(cfg)
    preds = inference(results, test_x)

    submit = pd.read_csv(path / cfg.output.submit)
    submit["prediction"] = preds

    today = str(date.today())
    if not (os.path.isdir(today)):
        os.makedirs(os.path.join(today))
        submit.to_csv(path / today / cfg.output.name, index=False)

    else:
        submit.to_csv(path / today / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
