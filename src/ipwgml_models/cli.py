import logging
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
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
@click.argument("target_config")
def train(
    model: str,
    reference_sensor: str,
    geometry: str,
    retrieval_input: Union[List[str], Dict[str, Any]],
    target_config: Union[Dict[str, Any]]
):
    """Run training for one of the IPWG baseline models. """

    try:
        mod = import_module(f"ipwgml_models.models.{model}")
    except ModuleNotFoundError:
        LOGGER.exception(
            f"Could not find a model with the name {model}."
        )
        return 1

    retrieval_input = eval(retrieval_input)
    if isinstance(retrieval_input, list):
        retrieval_input = [InputConfig.parse(inpt) for inpt in retrieval_input]
    else:
        retrieval_input = [InputConfig.parse(retrieval_input)]

    target_config = eval(target_config)
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
@click.argument("target_config")
@click.option("--n_processes", default=None)
def evaluate(
    model: str,
    domain: str,
    reference_sensor: str,
    geometry: str,
    retrieval_input: Union[List[str], Dict[str, Any]],
    target_config: Union[Dict[str, Any]],
    n_processes: Optional[int] = None
):
    """Run training for one of the IPWG baseline models. """

    try:
        mod = import_module(f"ipwgml_models.models.{model}")
    except ModuleNotFoundError:
        LOGGER.exception(
            f"Could not find a model with the name {model}."
        )
        return 1

    retrieval_input = eval(retrieval_input)
    if isinstance(retrieval_input, list):
        retrieval_input = [InputConfig.parse(inpt) for inpt in retrieval_input]
    else:
        retrieval_input = [InputConfig.parse(retrieval_input)]

    target_config = eval(target_config)
    target_config = TargetConfig(**target_config)

    evaluator = Evaluator(
        domain,
        reference_sensor,
        geometry,
        retrieval_input,
        target_config,
    )

    retrieval = mod.Retrieval(Path("."))
    evaluator.evaluate(
        retrieval_fn=retrieval,
        tile_size=retrieval.tile_size,
        overlap=retrieval.overlap,
        batch_size=retrieval.batch_size,
        input_data_format=retrieval.input_data_format,
        n_processes=n_processes
    )
    results = evaluator.get_results()
    results.to_netcdf(f'results_{domain}.nc')
