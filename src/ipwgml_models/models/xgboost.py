"""
ipwgml_models.models.xgboost
============================

This module provides functionality to train and evaluate an XGBoost regressor on the
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

INPUT_NAMES = {
    "gmi": ["obs_gmi"],
    "atms": ["obs_atms"],
    "geo_ir": ["obs_geo_ir"],
    "geo": ["obs_geo"],
    "ancillary": ["ancillary"],
}


def train(
    reference_sensor: str,
    geometry: str,
    retrieval_input: List[InputConfig],
    target_config: TargetConfig,
    output_path: Path
) -> None:
    """
    Training function for the iwpgml_models CLI.

    Args:
        reference_sensor: The reference sensor defining which subset of the SPR dataset to train the model on.
        geometry: The geometry to use for the retrieval: 'on_swath' or 'gridded'.
        retrieval_input: List of retrieval inputs to use.
        target_config: The configuration for the target data to use for training.
        output_path: The path to which to write the trained model data.
    """
    training_files = download_dataset(
        dataset_name="spr",
        reference_sensor=reference_sensor,
        input_data=retrieval_input,
        split="training",
        geometry=geometry,
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

    input_names = []
    for inpt in retrieval_input:
        input_names += INPUT_NAMES[inpt.name]
    model.input_names = input_names

    model.fit(x_train, y_train)



    with open(output_path / "model.pckl", "wb") as f:
        pickle.dump(model, f)


class Retrieval:
    """
    This class implements the interface to run the XGBoost retrieval on the SPR data.
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
        Run XGBoost retrieval on SPR input data.

        Args:
            input_data: An xarray.Dataset containing the input data.

        Return:
            An xarray.Dataset containing the retrieval results.
        """
        input_data = input_data.transpose("batch", ...)
        x = np.concatenate([input_data[name].data for name in self.model.input_names], axis=1)
        surface_precip = self.model.predict(x)
        return xr.Dataset({
            "surface_precip": (("samples",), surface_precip)
        })
