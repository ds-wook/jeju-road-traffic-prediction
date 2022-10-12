# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_parquet("../input/jeju-road-traffic-prediction/train.parquet")
test = pd.read_parquet("../input/jeju-road-traffic-prediction/test.parquet")

# %%
train.head()
# %%
train["maximum_speed_limit"].head()
# %%
train["height_restricted"].max()
# %%
train["road_name"].unique().shape
# %%
test["road_name"].unique().shape
# %%
set(test["road_name"].unique()) - set(train["road_name"].unique())
# %%
train["rush_hour"] = train["base_hour"].apply(
    lambda x: 1 if 8 <= x <= 10 or 17 <= x <= 19 else 0
)

# %%
train[["base_hour", "rush_hour"]]
# %%
