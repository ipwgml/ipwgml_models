"""
ipwgml_models.models.mlp
========================

This module implements a single-pixel retrieval for the IPWGML dataset based
on a multi-layer perceptron (MLP). The MLP is implemented using the
 'pytorch_retrieve' package.
"""
import dataclasses
from pathlib import Path
import subprocess
from toml import loads, dump
from typing import List
import sys

import torch
from pytorch_retrieve.architectures import load_model
import xarray as xr

from ipwgml.input import InputConfig
from ipwgml.target import TargetConfig


MODEL_CFG = """
name = "model"

[architecture]
name = "MLP"

[architecture.body]
hidden_channels = 512
n_layers = 8
residual_connections = "simple"
activation_factory = "GELU"
normalization_factory = "LayerNorm"
"""

def get_input(input_cfg: InputConfig) -> str:

    name = "obs_" + input_cfg.name
    if input_cfg.name == "ancillary":
        name = "ancillary"

    input_str = f"""

    [input.{name}]
    name = "{name}"
    n_features = {input_cfg.features[name]}
    normalize = "minmax"
    """
    return input_str


OUTPUT_CFG = """

[output.surface_precip]
kind = "Quantiles"
quantiles = 32
"""


TRAINING_CFG = """
[stage_1]
dataset_module = "ipwgml.pytorch.datasets"
training_dataset = "SPRTabular"
validation_split = 0.2
optimizer = "Adam"
optimizer_args = {lr=1e-3}
scheduler = "CosineAnnealingLR"
scheduler_args = {"T_max"=20}
n_epochs = 20
batch_size = 0
num_dataloader_workers = 1
metrics = ["MSE", "Bias", "CorrelationCoef"]

[stage_1.training_dataset_args]
batch_size = 2048
shuffle = true

[stage_1.validation_dataset_args]
batch_size = 2048
shuffle = true

[stage_2]
dataset_module = "ipwgml.pytorch.datasets"
training_dataset = "SPRTabular"
validation_split = 0.2
optimizer = "Adam"
optimizer_args = {lr=1e-3}
scheduler = "CosineAnnealingLR"
scheduler_args = {"T_max"=40}
n_epochs = 40
batch_size = 0
num_dataloader_workers = 1
metrics = ["MSE", "Bias", "CorrelationCoef"]

[stage_2.training_dataset_args]
batch_size = 2048
shuffle = true

[stage_2.validation_dataset_args]
batch_size = 2048
shuffle = true
"""

COMPUTE_CFG = """
accelerator="cuda"
"""


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
    model_cfg = loads(
        MODEL_CFG +
        "\n".join([get_input(inpt) for inpt in retrieval_input])
        + OUTPUT_CFG
    )

    training_cfg = loads(TRAINING_CFG)
    for stage in ["stage_1", "stage_2"]:
        training_cfg[f"{stage}"]["training_dataset_args"].update({
            "reference_sensor": reference_sensor,
            "split": "training",
            "geometry": geometry,
            "retrieval_input": [inpt.to_dict() for inpt in retrieval_input],
            "target_config": target_config.to_dict()
        })
        training_cfg[f"{stage}"]["validation_dataset_args"].update({
            "reference_sensor": reference_sensor,
            "split": "validation",
            "geometry": geometry,
            "retrieval_input": [inpt.to_dict() for inpt in retrieval_input],
            "target_config": target_config.to_dict()
        })

        print([inpt.to_dict() for inpt in retrieval_input])


    with open(output_path / "model.toml", "w") as output:
        dump(model_cfg, output)

    with open(output_path / "training.toml", "w") as output:
        dump(training_cfg, output)

    with open(output_path / "compute.toml", "w") as output:
        output.write(COMPUTE_CFG)

    subprocess.run(["pytorch_retrieve", "eda"], stderr=sys.stderr, stdout=sys.stdout)
    subprocess.run(["pytorch_retrieve", "train"], stderr=sys.stderr, stdout=sys.stdout)


class Retrieval:
    """

    """
    def __init__(
            self,
            model_path: Path
    ):
        """
        Args:
            model: A torch.nn.Module implementing the retrieval.
            retrieval_input: A list defining the retrieval input.
            precip_threshold: The probability threshold to apply to
                transform the 'probability_of_precip' to a 'precip_flag'
                output.
            heavy_precip_threshold: Same as 'precip_threshold' but for
                heavy precip flag output.
            stack: Whether or not the model expects the input data to
                be stacked ('True') or as dictionary.
            logits: Whether the model returns logits instead of probabilities.
            device: A torch.device defining the device on which to perform
                inference.
            dtype: The dtype to which to convert the retrieval input.
        """
        self.model = load_model(model_path / "model.pt")
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.dtype = torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()

        self.tile_size = None
        self.overlap = None
        self.input_data_format = "tabular"
        self.batch_size = 4096


    def __call__(self, input_data: xr.Dataset) -> xr.Dataset:
        """
        Run retrieval on input data.
        """
        dims = ("batch",)
        inpt = {}
        for name in self.model.input_config.keys():
            inpt_data = torch.tensor(input_data[name].data).to(self.device, self.dtype)
            inpt_data = inpt_data.transpose(0, 1)
            inpt[name] = inpt_data

        with torch.no_grad():
            pred = self.model(inpt)
            results = xr.Dataset()
            if "surface_precip" in pred:
                results["surface_precip"] = (
                    dims,
                    pred["surface_precip"].expected_value().select(1, 0).cpu().numpy()
                )

        return results
