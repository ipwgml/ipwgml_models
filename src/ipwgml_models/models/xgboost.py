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
        [
            next(iter(inpt.load_data(training_files[inpt.name][0], None).values()))
            for inpt in retrieval_input
        ],
        1
    )
    x_train = x_train.T
    y_train = target_config.load_reference_precip(
        training_files["target"][0]
    )
    valid = np.isfinite(y_train)
    x_train = x_train[valid]
    y_train = y_train[valid]

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

    with open(output_path / "model.pckl", "wb") as f:
        pickle.dump(model, f)


class Retrieval:
    """
    This class implements the actual XGBoos retrieval and provides the interface
    to evaluate the model on the IPWGML SPR dataset.
    """
    def __init__(self, path: Path):
        """
        Args:
            path: The path where the training of the model has been performed.
        """
        path = Path(path)
        model_path = path / "model.pckl"
        with open(model_path, "rb") as inpt:
            self.model = pickle.load(inpt)

        self.tile_size = None
        self.overlap = None
        self.input_data_format = "tabular"
        self.batch_size = 4096


    def __call__(self, input_data: xr.Dataset) -> xr.Dataset:
        """
        This function implements the retrieval inference for the XBGBoost model.

        Args:
            input_data: An xarray.Dataset containing the input data.

        Return:
            An xarray.Dataset containing the retrieval results.
        """
        input_data = input_data.transpose("batch", ...)
        obs = input_data["obs_gmi"].data
        sp = self.model.predict(obs)
        return xr.Dataset({
            "surface_precip": (("samples",), sp)
        })
