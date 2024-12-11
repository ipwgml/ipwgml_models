# IPWGM ML models

This repository collects machine-learning models for the IPWG Satellite Precipitation Retrieval (SPR) machine-learning benchmark dataset.

## Overview

This repostiory collects code to access and evaluate precipitation retrieval on the SPR dataset. We welcome two kinds or contributions:

 - Internal models: Internal models generic that can be trained using the ``ipwgml_models`` package on any combination of the input data included in the SPR dataset.
 - External models: External models can only be evaluated. 


## Submitting models

To submit a model please open a pull request containing the code necessary to train and/or evaluate the model using the ``ipwgml_models`` package.

### Internal models

Internal models should be added in a separate module file in ``src/ipwgml_models/models``. Each module must implement a ``train`` function and a ``Retrieval`` class providing the interfaces for training and evaluation of the model, respectively.

### External models

External models should be provided in the ``external`` folder. Each folder should contain a ``__init__.py`` making it importable as a module. The module should provide a retrieval class that can be used to evaluate the model.
