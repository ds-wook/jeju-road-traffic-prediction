# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_parquet("../input/jeju-road-traffic-prediction/train.parquet")
test = pd.read_parquet("../input/jeju-road-traffic-prediction/test.parquet")

# %%
train["maximum_speed_limit"].head()
# %%
train["height_restricted"].max()
# %%
