# IPWGM ML models

This repository collects machine-learning models for the IPWG Satellite Precipitation Retrieval (SPR) machine-learning benchmark dataset.

## Overview

This repostiory provides code to access and evaluate precipitation retrieval on the SPR dataset. We welcome two kinds or contributions:

 - Internal models: Internal models are generic in the sense that they can be trained on any combination of the input data included in the SPR dataset using the interface provided by the ``ipwgml_models`` package..
 - External models: External models are models that were trained externally and can only be evaluated.


## Submitting models

To submit a model please open a pull request containing the code necessary to train and/or evaluate the model using the ``ipwgml_models`` package.

### Internal models

Internal models should be added in a separate module file in ``src/ipwgml_models/models``. Each module must implement a ``train`` function and a ``Retrieval`` class providing the interfaces for training and evaluation of the model, respectively.

### External models

External models should be provided in the ``external`` folder. Each folder should contain a ``__init__.py`` making it importable as a Python module. The module should provide a retrieval class that can be used to evaluate the model.

## Installation

The ``ipwgml_models`` package currently relies on the ``pytorch_retrieve`` and ``ipwgml`` packages. The three packages can be install using the following command:

``` shellsession
pip install git+https://github.com/simonpf/pytorch_retrieve git+https://github.com/ipwgml/ipwgml git+https://github.com/ipwgml/ipwgml_models
```
