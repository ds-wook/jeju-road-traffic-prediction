from typing import NoReturn

import pandas as pd
from omegaconf import DictConfig, open_dict
from optuna.trial import FrozenTrial
from sklearn.metrics import mean_absolute_error

from models.boosting import LightGBMTrainer, XGBoostTrainer
from tuning.base import BaseTuner


class LightGBMTuner(BaseTuner):
    def __init__(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        config: DictConfig,
    ) -> NoReturn:
        self.train_x = train_x
        self.train_y = train_y
        super().__init__(config)

    def _objective(self, trial: FrozenTrial) -> float:
        """
        Objective function
        Args:
            trial: trial object
            config: config object
        Returns:
            metric score
        """
        # trial parameters
        params = {
            "objective": self.config.tuning.params.objective,
            "verbose": self.config.tuning.params.verbose,
            "boosting_type": self.config.tuning.params.boosting_type,
            "n_jobs": self.config.tuning.params.n_jobs,
            "seed": self.config.tuning.params.seed,
            "learning_rate": trial.suggest_float(
                "learning_rate", *self.config.tuning.params.learning_rate
            ),
            "lambda_l1": trial.suggest_loguniform(
                "lambda_l1", *self.config.tuning.params.lambda_l1
            ),
            "lambda_l2": trial.suggest_loguniform(
                "lambda_l2", *self.config.tuning.params.lambda_l2
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", *self.config.tuning.params.num_leaves
            ),
            "feature_fraction": trial.suggest_uniform(
                "feature_fraction", *self.config.tuning.params.feature_fraction
            ),
            "bagging_fraction": trial.suggest_uniform(
                "bagging_fraction", *self.config.tuning.params.bagging_fraction
            ),
            "bagging_freq": trial.suggest_int(
                "bagging_freq", *self.config.tuning.params.bagging_freq
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", *self.config.tuning.params.min_data_in_leaf
            ),
        }

        # config update
        with open_dict(self.config.models):
            self.config.models.params.update(params)

        lgbm_trainer = LightGBMTrainer(self.config)
        lgbm_results = lgbm_trainer.train_cross_validation(self.train_x, self.train_y)

        scores = mean_absolute_error(self.train_y, lgbm_results.oof_preds)

        return scores


class XGBoostTuner(BaseTuner):
    def __init__(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        config: DictConfig,
    ) -> NoReturn:
        self.train_x = train_x
        self.train_y = train_y
        super().__init__(config)

    def _objective(self, trial: FrozenTrial) -> float:
        """
        Objective function
        Args:
            trial: trial object
            config: config object
        Returns:
            metric score
        """
        # trial parameters
        params = {
            "tree_method": self.config.tuning.params.tree_method,
            "objective": self.config.tuning.params.objective,
            "eval_metric": self.config.tuning.params.eval_metric,
            "num_class": self.config.tuning.params.num_class,
            "seed": self.config.tuning.params.seed,
            "eta": trial.suggest_float("eta", *self.config.tuning.params.eta),
            "reg_alpha": trial.suggest_loguniform(
                "reg_alpha", *self.config.tuning.params.reg_alpha
            ),
            "reg_lambda": trial.suggest_loguniform(
                "reg_lambda", *self.config.tuning.params.reg_lambda
            ),
            "max_leaves": trial.suggest_int(
                "max_leaves", *self.config.tuning.params.max_leaves
            ),
            "gamma": trial.suggest_loguniform(
                "gamma", *self.config.tuning.params.gamma
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *self.config.tuning.params.min_child_weight
            ),
            "max_depth": trial.suggest_int(
                "max_depth", *self.config.tuning.params.max_depth
            ),
            "colsample_bytree": trial.suggest_uniform(
                "colsample_bytree", *self.config.tuning.params.colsample_bytree
            ),
            "subsample": trial.suggest_uniform(
                "subsample", *self.config.tuning.params.subsample
            ),
        }

        # config update
        with open_dict(self.config.models):
            self.config.models.params.update(params)

        xgb_trainer = XGBoostTrainer(self.config)
        xgb_trainer.train_cross_validation(self.train_x, self.train_y)

        score = mean_absolute_error(self.train_y, xgb_trainer.oof_preds)

        return score
