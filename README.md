# Variational Hydrology

Code for the paper: *A variational approach at uncertainty estimation in rainfall-runoff modeling*.

## Installation
Installation is not so easy because some dependency issues between `numpy`, `numba` and `llvmlite`. My suggestion is to install using [uv](https://docs.astral.sh/uv/) or carefully follow the list of requirements in the `pyproject.toml`.

Here's the command to install using `uv` with the project directory being the current working folder:
```console
uv sync
```

Check the requirements in the `pyproject.toml`.

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
