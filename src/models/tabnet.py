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
        params = dict(self.config.models.params)
        params["cat_idxs"] = []
        params["cat_dims"] = []
        params["optimizer_fn"] = torch.optim.Adam
        params["optimizer_params"] = dict(lr=2e-2, weight_decay=1e-5)
        model = TabNetRegressor(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            eval_name=["train", "val"],
            patience=200,
            max_epochs=self.config.models.max_epochs,
            batch_size=self.config.models.batch_size,
            virtual_batch_size=self.config.models.virtual_batch_size,
            num_workers=0,
            drop_last=False,
            eval_metric=["mae"],
        )

        return model
