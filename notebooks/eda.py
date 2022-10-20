# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_parquet("../input/jeju-road-traffic-prediction/train.parquet")
test = pd.read_parquet("../input/jeju-road-traffic-prediction/test.parquet")


# %%
train["group_node_name"] = train["start_node_name"] + "_" + train["end_node_name"]
test["group_node_name"] = test["start_node_name"] + "_" + test["end_node_name"]
# %%
group_node = list(set(train["group_node_name"]) & set(test["group_node_name"]))
train[train["group_node_name"].isin(group_node)].head()
# %%
train[train["group_node_name"].isin(group_node)].shape
# %%
test[test["group_node_name"].isin(group_node)].shape
# %%
test.shape
# %%
from omegaconf import OmegaConf

features = OmegaConf.load("../config/features/engineering.yaml")
features
# %%
features["selected_features"] = group_node
# %%
OmegaConf.save(features, "../config/features/selected_features.yaml")
# %%
