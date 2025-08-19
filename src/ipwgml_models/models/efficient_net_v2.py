#!/usr/bin/env python3
from pathlib import Path
import subprocess
from toml import loads, dump
from typing import Any, Dict, List, Optional
import sys

import torch
from pytorch_retrieve.architectures import load_model
import xarray as xr

from ipwgml.input import InputConfig
from ipwgml.target import TargetConfig
from ipwgml.pytorch import PytorchRetrieval


INPUTS = {
    "obs_gmi": "gmi",
    "obs_atms": "atms",
    "ancillary": "ancillary",
    "obs_geo_ir": "geo_ir",
    "obs_geo": "geo_ir",
}

MODEL_CFG = """
name = "ipwgml_efficient_net_v2"
[architecture]
name = "EncoderDecoder"
[architecture.encoder]
channels = [24, 48, 64, 128, 160, 256]
downsampling_factors = [1, 2, 2, 2, 2]
stage_depths = [1, 2, 2, 3, 4, 15]
block_factory = [
    "InvertedBottleneck",
    "InvertedBottleneck",
    "InvertedBottleneck",
    "InvertedBottleneck",
    "InvertedBottleneck",
    "InvertedBottleneck",
]
block_factory_args = [
    {activation_factory="GELU", normalization_factory="LayerNormFirst", expansion_factor=1, excitation_ratio=0.0, fused=true, anti_aliasing=true},
    {activation_factory="GELU", normalization_factory="LayerNormFirst", stochastic_depth=0.9, expansion_factor=4, excitation_ratio=0.0, fused=true, anti_aliasing=true},
    {activation_factory="GELU", normalization_factory="LayerNormFirst", stochastic_depth=0.8, expansion_factor=4, excitation_ratio=0.0, fused=true, anti_aliasing=true},
    {activation_factory="GELU", normalization_factory="LayerNormFirst", stochastic_depth=0.7, expansion_factor=4, excitation_ratio=0.25, anti_aliasing=true},
    {activation_factory="GELU", normalization_factory="LayerNormFirst", stochastic_depth=0.6, expansion_factor=6, excitation_ratio=0.25, anti_aliasing=true},
    {activation_factory="GELU", normalization_factory="LayerNormFirst", stochastic_depth=0.5, expansion_factor=6, excitation_ratio=0.25, anti_aliasing=true},
]

[architecture.decoder]
channels = [160, 128, 64, 48]
upsampling_factors = [2, 2, 2, 2]
stage_depths = [4, 3, 2, 3]
block_factory = [
    "InvertedBottleneck",
    "InvertedBottleneck",
    "InvertedBottleneck",
    "InvertedBottleneck",
]
block_factory_args = [
    {activation_factory="GELU", normalization_factory="LayerNormFirst", stochastic_depth=0.6, expansion_factor=6, excitation_ratio=0.25, anti_aliasing=true},
    {activation_factory="GELU", normalization_factory="LayerNormFirst", stochastic_depth=0.7, expansion_factor=4, excitation_ratio=0.25, anti_aliasing=true},
    {activation_factory="GELU", normalization_factory="LayerNormFirst", stochastic_depth=0.8, expansion_factor=4, excitation_ratio=0.0, fused=true, anti_aliasing=true},
    {activation_factory="GELU", normalization_factory="LayerNormFirst", stochastic_depth=1.0, expansion_factor=4, fused=true, anti_aliasing=true},
]
skip_connections=true

[architecture.stem]
individual = false
depth = 1
in_channels = 16
out_channels = 24
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
training_dataset = "SPRSpatial"
optimizer = "Adam"
optimizer_args = {lr=1e-3}
scheduler = "CosineAnnealingLR"
scheduler_args = {"T_max"=20}
n_epochs = 20
batch_size = 16
n_dataloader_workers = 8
metrics = ["MSE", "Bias", "CorrelationCoef"]

[stage_1.training_dataset_args]
augment = true

[stage_1.validation_dataset_args]
augment = false

[stage_2]
dataset_module = "ipwgml.pytorch.datasets"
training_dataset = "SPRSpatial"
optimizer = "Adam"
optimizer_args = {lr=5e-4}
scheduler = "CosineAnnealingLR"
scheduler_args = {"T_max"=40}
n_epochs = 40
batch_size = 16
n_dataloader_workers = 8
metrics = ["MSE", "Bias", "CorrelationCoef"]

[stage_2.training_dataset_args]
augment = true

[stage_2.validation_dataset_args]
augment = false
"""

COMPUTE_CFG = """
accelerator="{accelerator}"
devices={devices}
"""

def train(
    reference_sensor: str,
    geometry: str,
    retrieval_input: List[InputConfig],
    target_config: TargetConfig,
    output_path: Path,
    options: Optional[Dict[str, Any]] = None
):
    """
    Training function for the iwpgml_models CLI.
    """

    if options is None:
        options = {}

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


    with open(output_path / "model.toml", "w") as output:
        dump(model_cfg, output)

    with open(output_path / "training.toml", "w") as output:
        dump(training_cfg, output)

    devices = options.get("devices", "[0]")
    accelerator = options.get("accelerator", "cuda")
    compute_cfg = COMPUTE_CFG.format(devices=devices, accelerator=accelerator)

    with open(output_path / "compute.toml", "w") as output:
        output.write(compute_cfg)

    subprocess.run(["pytorch_retrieve", "eda"], stderr=sys.stderr, stdout=sys.stdout)
    subprocess.run(["pytorch_retrieve", "train"], stderr=sys.stderr, stdout=sys.stdout)


class Retrieval(PytorchRetrieval):
    """

    """
    def __init__(
        self,
        model_path: Path
    ):
        model = load_model(model_path / "model.pt")

        super().__init__(
            model,
            [INPUTS[name] for name in model.input_config.keys()],
        )

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.dtype = torch.float32
        self.model = self.model.to(device=self.device).eval()

        self.tile_size = (256, 256)
        self.overlap = 64
        self.input_data_format = "spatial"
        self.batch_size = 8

    def __call__(self, input_data: xr.Dataset) -> xr.Dataset:
        """
        Run retrieval on input data.
        """
        input_data = input_data.transpose("batch", "channels_gmi", "latitude", "longitude")
        dims = ("batch", "latitude", "longitude")
        inpt = {}
        for name in self.model.input_config.keys():
            inpt_data = torch.tensor(input_data[name].data).to(self.device, self.dtype)
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
