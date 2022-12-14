from typing import NoReturn

import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_tabnet.tab_model import TabNetRegressor

from models.base import BaseModel


class TabNetTrainer(BaseModel):
    def __init__(self, config: DictConfig) -> NoReturn:
        super().__init__(config)

    def _train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> TabNetRegressor:
        x_train, y_train = x_train.values, y_train.values.reshape(-1, 1)
        x_valid, y_valid = x_valid.values, y_valid.values.reshape(-1, 1)

        model = TabNetRegressor(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=self.config.models.params.lr),
            scheduler_params={
                "step_size": self.config.models.params.step_size,
                "gamma": self.config.models.params.gamma,
            },
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type=self.config.models.params.mask_type,
            n_steps=self.config.models.params.n_steps,
            n_d=self.config.models.params.n_d,
            n_a=self.config.models.params.n_a,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            eval_name=["train", "val"],
            max_epochs=self.config.models.max_epochs,
            batch_size=self.config.models.batch_size,
            virtual_batch_size=self.config.models.virtual_batch_size,
            patience=2,
            num_workers=1,
            drop_last=False,
            eval_metric=["mae"],
        )

        return model
