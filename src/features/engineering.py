import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from category_encoders.target_encoder import TargetEncoder
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

tqdm.pandas()


def create_categorical_train(train: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        config: config
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.data.encoder

    le_encoder = LabelEncoder()

    for cat_feature in tqdm(config.features.cat_features):
        train[cat_feature] = le_encoder.fit_transform(train[cat_feature])
        with open(path / f"{cat_feature}.pkl", "wb") as f:
            pickle.dump(le_encoder, f)

    return train


def create_categorical_test(test: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        config: config
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.data.encoder

    for cat_feature in tqdm(config.features.cat_features):
        le_encoder = pickle.load(open(path / f"{cat_feature}.pkl", "rb"))
        for label in np.unique(test[cat_feature]):
            if label not in le_encoder.classes_:
                le_encoder.classes_ = np.append(le_encoder.classes_, label)
        test[cat_feature] = le_encoder.transform(test[cat_feature])

    return test


def haversine_array(
    start_lat: pd.Series, start_lng: pd.Series, end_lat: pd.Series, end_lng: pd.Series
) -> pd.Series:
    start_lat, start_lng, end_lat, end_lng = map(
        np.radians, (start_lat, start_lng, end_lat, end_lng)
    )
    avg_earth_radius = 6371
    lat = end_lat - start_lat
    lng = end_lng - start_lng
    d = (
        np.sin(lat * 0.5) ** 2
        + np.cos(start_lat) * np.cos(end_lat) * np.sin(lng * 0.5) ** 2
    )
    h = 2 * avg_earth_radius * np.arcsin(np.sqrt(d))
    return h


def change_target_encoding(
    df: pd.DataFrame, config: DictConfig, is_train=True
) -> pd.DataFrame:
    """
    Change target encoding
    Args:
        df: dataframe
        config: config
        is_train: is train
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.data.encoder

    df["group_node"] = df["start_node_name"] + "_" + df["end_node_name"]
    df["group_node"] = df["group_node"].astype("category")

    if is_train:
        target_encoder = TargetEncoder(cols=["group_node"])
        target_encoder.fit(df["group_node"], df["target"])
        df["group_node"] = target_encoder.transform(df["group_node"])

        with open(path / "group_node.pkl", "wb") as f:
            pickle.dump(target_encoder, f)

    else:
        target_encoder = pickle.load(open(path / "group_node.pkl", "rb"))
        df["group_node"] = target_encoder.transform(df["group_node"])

    return df


def add_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add kmeans features
    Args:
        df: dataframe
    Returns:
        dataframe
    """
    kmeans = KMeans(n_clusters=32, random_state=42)
    kmeans.fit(df[["start_latitude", "start_longitude"]])
    df["start_cluster"] = kmeans.predict(df[["start_latitude", "start_longitude"]])

    kmeans = KMeans(n_clusters=32, random_state=42)
    kmeans.fit(df[["end_latitude", "end_longitude"]])
    df["end_cluster"] = kmeans.predict(df[["end_latitude", "end_longitude"]])

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features
    Args:
        df: dataframe
    Returns:
        dataframe
    """
    # add group node
    df["group_node"] = df["start_node_name"] + "_" + df["end_node_name"]
    df["group_node"] = df["group_node"].astype("category")
    # add haversine distance
    df["distance"] = haversine_array(
        df["start_latitude"],
        df["start_longitude"],
        df["end_latitude"],
        df["end_longitude"],
    )

    return df
