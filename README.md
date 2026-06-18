# Variational Hydrology
[![Identifier]](https://img.shields.io/badge/doi-10.18419%2Fdarus--5118-d45815.svg)](https://doi.org/10.18419/darus-5118)

Code for the paper: *A variational approach at uncertainty estimation in rainfall-runoff modeling*.

## Installation
Installation is not so easy because some dependency issues between `numpy`, `numba` and `llvmlite`. My suggestion is to install using [uv](https://docs.astral.sh/uv/) or carefully follow the list of requirements in the `pyproject.toml`.

Here's the command to install using `uv` with the project directory being the current working folder:
```console
uv sync
```

Check the requirements in the `pyproject.toml`.

### Dependency issues
This code depends on an older version of the [Hy2DL](https://github.com/eduardoAcunaEspinoza/Hy2DL) library (0.2.0). This is included as a submodule in the `root` directory in the DaRUS repository.

## Usage

In the [examples](./examples/) directory there's an example for how to code a model inside a Jupyer Notebook. Typically a model will get coded before adapting an existing training script in the [scripts](./scripts/) directory to train using an external GPU and more data.

To run a script in Windows:

```console
source .venv/Scripts/activate
python scripts/train_vlstm.py
```
In Linux switch *Scripts* for *bin*. Alternatively `uv` can directly run a script:

```console
uv run scripts/train_vlstm.py
```
