from typing import NoReturn

import lightgbm as lgb
import pandas as pd
import wandb.catboost as wandb_cb
import wandb.lightgbm as wandb_lgb
import wandb.xgboost as wandb_xgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from omegaconf import DictConfig

from models.base import BaseModel


class LightGBMTrainer(BaseModel):
    def __init__(self, config: DictConfig) -> NoReturn:
        super().__init__(config)

    def _train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> lgb.Booster:
        train_set = lgb.Dataset(
            x_train, y_train
        )
        valid_set = lgb.Dataset(
            x_valid, y_valid
        )

        model = lgb.train(
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            params=dict(self.config.models.params),
            verbose_eval=self.config.models.verbose_eval,
            num_boost_round=self.config.models.num_boost_round,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            callbacks=[wandb_lgb.wandb_callback()]
            if self.config.log.experiment
            else None,
        )

        return model


class CatBoostTrainer(BaseModel):
    def __init__(self, config: DictConfig) -> NoReturn:
        super().__init__(config)

    def _train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> CatBoostRegressor:

        train_set = Pool(
            x_train, y_train, cat_features=self.config.features.cat_features
        )
        valid_set = Pool(
            x_valid, y_valid, cat_features=self.config.features.cat_features
        )

        model = CatBoostRegressor(
            random_state=self.config.models.seed,
            cat_features=self.config.features.cat_features,
            task_type=self.config.models.task_type,
            **self.config.models.params,
        )
        model.fit(
            train_set,
            eval_set=valid_set,
            verbose_eval=self.config.models.verbose_eval,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            callbacks=[wandb_cb.WandbCallback()]
            if self.config.models.task_type == "CPU" and self.config.log.experiment
            else None,
        )

        return model


class XGBoostTrainer(BaseModel):
    def __init__(self, config: DictConfig) -> NoReturn:
        super().__init__(config)

    def _train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> xgb.Booster:
        dtrain = xgb.DMatrix(x_train, y_train)
        dvalid = xgb.DMatrix(x_valid, y_valid)
        watchlist = [(dtrain, "train"), (dvalid, "eval")]

        model = xgb.train(
            dtrain=dtrain,
            evals=watchlist,
            params=dict(self.config.models.params),
            num_boost_round=self.config.models.num_boost_round,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            verbose_eval=self.config.models.verbose_eval,
            callbacks=[wandb_xgb.WandbCallback()]
            if self.config.log.experiment
            else None,
        )

        return model
