"""
ipwgml_models.cli
=================

The command line interface for the 'ipwgml_models' package.
"""
import logging
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import sys

import click
import xarray as xr

from ipwgml.input import InputConfig
from ipwgml.evaluation import Evaluator
from ipwgml.target import TargetConfig


LOGGER = logging.getLogger(__name__)


@click.group
def ipwgml_models():
    """Command line interface for the 'ipwgml' package."""

@ipwgml_models.command(name="train")
@click.argument("model")
@click.argument("reference_sensor")
@click.argument("geometry")
@click.argument("retrieval_input")
@click.argument("target_config", default=None, required=False)
def train(
    model: str,
    reference_sensor: str,
    geometry: str,
    retrieval_input: Union[List[str], Dict[str, Any]],
    target_config: Optional[Union[Dict[str, Any]]] = None
):
    """
    Train MODEL on REFERENCE_SENSOR observations for given geometry, retrieval input, and target configuration.

    MODEL: The name of the model to train.

    REFERENCE_SENSOR: The name of the reference sensor defining which subset of
    the IPWG SPR dataset to use for training: 'gmi' or 'atms'.

    GEOMETRY: The geometry to use for the training: 'on_swath' or 'gridded'.

    RETRIEVAL INPUT: An evaluatable Python string defining the retrieval input for the SPR dataset.
    Example: "[{'name': 'gmi', 'normalize': 'minmax'}]"

    TARGET_CONFIG: An evaluatable Python string defining the target config for the SPR dataset.
    Example: "[{'min_rqi': 0.5}]"
    """
    try:
        mod = import_module(f"ipwgml_models.models.{model}")
    except ModuleNotFoundError:
        LOGGER.exception(
            f"Could not find a model with the name {model}."
        )
        return 1

    if retrieval_input[0] not in ["'", '"']:
        retrieval_input = f"'{retrieval_input}'"
    retrieval_input = eval(retrieval_input)
    if isinstance(retrieval_input, list):
        retrieval_input = [InputConfig.parse(inpt) for inpt in retrieval_input]
    else:
        retrieval_input = [InputConfig.parse(retrieval_input)]

    if target_config is not None:
        target_config = eval(target_config)
    else:
        target_config = {}
    target_config = TargetConfig(**target_config)

    mod.train(
        reference_sensor,
        geometry,
        retrieval_input,
        target_config,
        Path(".")
    )


@ipwgml_models.command(name="evaluate")
@click.argument("model")
@click.argument("domain")
@click.argument("reference_sensor")
@click.argument("geometry")
@click.argument("retrieval_input")
@click.argument("target_config", default=None, required=False)
@click.option("--n_processes", default=None)
def evaluate(
    model: str,
    domain: str,
    reference_sensor: str,
    geometry: str,
    retrieval_input: Union[List[str], Dict[str, Any]],
    target_config: Optional[Union[Dict[str, Any]]] = None,
    n_processes: Optional[int] = None
):
    """
    Evaluated the trained MODEL on SPR evaluation data over DOMAIN using given geometry, retrieval input, and
    optional target configuration.

    Note: This comman should be executed from the same folder that the training command was issued.

    DOMAIN: The name of the evaluation domain: 'austria', 'korea', 'conus' .

    REFERENCE_SENSOR: The name of the reference sensor defining which subset of
    the IPWG SPR dataset to use for training: 'gmi' or 'atms'.

    GEOMETRY: The geometry to use for the training: 'on_swath' or 'gridded'.

    RETRIEVAL INPUT: An evaluatable Python string defining the retrieval input for the SPR dataset.
    Example: "[{'name': 'gmi', 'normalize': 'minmax'}]"

    TARGET_CONFIG: An evaluatable Python string defining the target config for the SPR dataset.
    Example: "[{'min_rqi': 0.5}]"
    """
    try:
        mod = import_module(f"ipwgml_models.models.{model}")
    except ModuleNotFoundError:
        mod = None
        pass

    if mod is None:
        try:
            sys.path.insert(0, ".")
            print(sys.path)
            mod = import_module(f"{model}")
        except ModuleNotFoundError:
            LOGGER.exception(
                f"Could not impoart an external model with the name {model}."
            )
            return 1

    if retrieval_input[0] not in ["'", '"']:
        retrieval_input = f"'{retrieval_input}'"
    retrieval_input = eval(retrieval_input)
    if isinstance(retrieval_input, list):
        retrieval_input = [InputConfig.parse(inpt) for inpt in retrieval_input]
    else:
        retrieval_input = [InputConfig.parse(retrieval_input)]

    if target_config is None:
        target_config = {}
    else:
        target_config = eval(target_config)
    target_config = TargetConfig(**target_config)


    retrieval = mod.Retrieval(Path("."))

    domains = domain.split(",")
    results = []
    for domain in domains:
        LOGGER.info(f"Starting evaluation over domain '{domain}'.")
        evaluator = Evaluator(
            reference_sensor,
            geometry,
            retrieval_input,
            domain=domain,
            target_config=target_config,
        )
        evaluator.evaluate(
            retrieval_fn=retrieval,
            tile_size=retrieval.tile_size,
            overlap=retrieval.overlap,
            batch_size=retrieval.batch_size,
        input_data_format=retrieval.input_data_format,
            n_processes=n_processes
        )
        res_d = evaluator.get_results()
        res_d["domain"] = domain
        results.append(res_d)

    results = xr.concat(results, dim="domain")
    results.to_netcdf(f'results.nc')
