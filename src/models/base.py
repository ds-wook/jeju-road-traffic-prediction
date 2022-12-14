import gc
import logging
import pickle
import warnings
from abc import ABCMeta, abstractclassmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, NoReturn, Union

import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: Dict[str, Any]
    scores: Dict[str, Union[float, Dict[str, float]]]


class BaseModel(metaclass=ABCMeta):
    def __init__(self, config: DictConfig) -> NoReturn:
        self.config = config
        self.result = None

    @abstractclassmethod
    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> NoReturn:
        raise NotImplementedError

    def save_model(self) -> NoReturn:
        """
        Save model
        """
        model_path = (
            Path(get_original_cwd())
            / self.config.models.path
            / self.config.models.working
            / self.config.models.result
        )

        with open(model_path, "wb") as output:
            pickle.dump(self.result, output)

    def train_cross_validation(
        self, train_x: pd.DataFrame, train_y: pd.DataFrame
    ) -> ModelResult:
        models = dict()
        scores = dict()

        kfold = KFold(
            n_splits=self.config.models.n_splits,
            random_state=self.config.data.seed,
            shuffle=True,
        )
        splits = kfold.split(train_x, train_y)
        oof_preds = np.zeros((train_x.shape[0],))

        for fold, (train_idx, valid_idx) in enumerate(splits, 1):

            x_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
            x_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

            if self.config.log.experiment:
                wandb.init(
                    entity=self.config.log.entity,
                    project=self.config.log.project,
                    name=self.config.log.name + f"-fold-{fold}",
                )
                model = self._train(x_train, y_train, x_valid, y_valid)
                wandb.finish()

            else:
                model = self._train(x_train, y_train, x_valid, y_valid)

            oof_preds[valid_idx] = (
                model.predict(xgb.DMatrix(x_valid))
                if isinstance(model, xgb.Booster)
                else model.predict(x_valid.values).flatten()
                if isinstance(model, TabNetRegressor)
                else model.predict(x_valid)
            )

            score = mean_absolute_error(y_valid, oof_preds[valid_idx])
            logging.info(f"Fold {fold} score: {score:.4f}")
            models[f"fold_{fold}"] = model
            scores[f"fold_{fold}"] = score

            del x_train, y_train, x_valid, y_valid, model
            gc.collect()

        oof_score = mean_absolute_error(train_y, oof_preds)
        logging.info(f"OOF Score: {oof_score}")

        self.result = ModelResult(
            oof_preds=oof_preds,
            models=models,
            scores={"oof_score": oof_score, "KFold_scores": scores},
        )

        return self.result
