import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from models.boosting import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    train_x, train_y = load_train_dataset(cfg)

    if cfg.models.working == "lightgbm":
        lgb_trainer = LightGBMTrainer(config=cfg)
        lgb_trainer.train_cross_validation(train_x, train_y)
        lgb_trainer.save_model()

    elif cfg.models.working == "xgboost":
        xgb_trainer = XGBoostTrainer(config=cfg)
        xgb_trainer.train_cross_validation(train_x, train_y)
        xgb_trainer.save_model()

    elif cfg.models.working == "catboost":
        cb_trainer = CatBoostTrainer(config=cfg)
        cb_trainer.train_cross_validation(train_x, train_y)
        cb_trainer.save_model()

    else:
        raise NotImplementedError


if __name__ == "__main__":
    _main()
