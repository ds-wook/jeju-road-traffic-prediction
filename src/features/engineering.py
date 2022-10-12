import pickle
from pathlib import Path

import numpy as np
import pandas as pd
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

    for cat_feature in tqdm(config.data.cat_features):
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

    for cat_feature in tqdm(config.data.cat_features):
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
    count_features = [
        "base_hour",
        "road_rating",
        "connect_code",
        "maximum_speed_limit",
        "weight_restricted",
        "road_type",
        "start_cluster",
        "end_cluster",
    ]
    for feature in count_features:
        df[feature] = df[feature].astype(int)
    # add group node
    df["group_node_name"] = df["start_node_name"] + "_" + df["end_node_name"]
    df["group_node_name"] = df["group_node_name"].astype("category")

    # add haversine distance
    df["distance"] = haversine_array(
        df["start_latitude"],
        df["start_longitude"],
        df["end_latitude"],
        df["end_longitude"],
    )

    return df
