"""
ipwgml_models.models.xgboost
============================

This module provides functionality to train an XGBoost regressor on the
IPWG ML SPR dataset.
"""
from pathlib import Path
import pickle
from typing import List
from ipwgml.data import download_dataset
from ipwgml.input import InputConfig
from ipwgml.target import TargetConfig
import numpy as np
import xgboost as xgb
import xarray as xr


def train(
    reference_sensor: str,
    geometry: str,
    retrieval_input: List[InputConfig],
    target_config: TargetConfig,
    output_path: Path
):
    """
    Training function for the iwpgml_models CLI.
    """
    training_files = download_dataset(
        dataset_name="spr",
        reference_sensor="gmi",
        input_data=["gmi"],
        split="training",
        geometry="on_swath",
        format="tabular"
    )
    x_train = np.concatenate(
        [inpt.load_data(training_files[inpt.name][0], None) for inpt in retrieval_input],
        1
    )
    y_train = target_config.load_reference_precip(
        training_files["target"]
    )

    # Define XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(x_train, y_train)

    with open(output_path / "model", "wb") as f:
        pickle.dump(model, f)


class Retrieval:
    def __init__(self, path: Path):
        pass
