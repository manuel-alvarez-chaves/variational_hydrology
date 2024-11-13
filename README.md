# Information Hydrology

A repository for the development of hydrological rainfall-runoff models using Information Theory.

## Installation
Installation is not so easy because some dependency issues between `numpy`, `numba` and `llvmlite`. My suggestion is to install using `poetry` or carefully follow the list of requirements in the `pyproject.toml`.

Here's an example for Windows:
```console
python -m venv .venv
source .venv/Scripts/activate

(.venv) pip install poetry
(.venv) poetry install
```
For Linux switch "Scripts" in the command beginning with source for "bin". So: 
```console
source .venv/bin/activate
```

## Usage

In the [examples](./examples/) directory there's an example for how to code a model inside a Jupyer Notebook. Typically a model will get coded before adapting an existing training script in the [scripts](./scripts/) directory to train using an external GPU and more data.