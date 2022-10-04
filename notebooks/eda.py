# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_parquet("../input/jeju-road-traffic-prediction/train.parquet")
test = pd.read_parquet("../input/jeju-road-traffic-prediction/test.parquet")

# %%
train["group_node_name"] = train["start_node_name"] + "_" + train["end_node_name"]
test["group_node_name"] = test["start_node_name"] + "_" + test["end_node_name"]

# %%
set(test["group_node_name"].unique()) - set(train["group_node_name"].unique())
# %%
test.shape
# %%
train.shape
# %%


sns.histplot(train["target"])
plt.show()
# %%
np.log1p(train["target"]).hist()
# %%
train.head()
# %%
sns.histplot(train["base_hour"])
plt.show()
# %%
