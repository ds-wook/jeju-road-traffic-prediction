from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.dataset import load_test_dataset
from models.infer import inference, load_model


@hydra.main(config_path="../config/", config_name="predict", version_base="1.2.0")
def _main(cfg: DictConfig):
    # model load
    results = load_model(cfg, cfg.models.result)

    test_x = load_test_dataset(cfg)
    preds = inference(results, test_x)

    submit = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.output.submit)
    submit[cfg.data.target] = preds

    # save
    path = Path(get_original_cwd()) / cfg.output.path
    submit.to_csv(path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
