# Information Hydrology

A repository for the development of hydrological rainfall-runoff models using Information Theory.

## Installation
Installation is not so easy because some dependency issues between `numpy`, `numba` and `llvmlite`. My suggestion is to install using [uv](https://docs.astral.sh/uv/) or carefully follow the list of requirements in the `pyproject.toml`.

Here's an example with the project directory being the current working folder:
```console
uv sync
```
This project also benefits of having local copies of `neuralhydrology` and `unite_toolbox`. If these projects are at the same level of the `information_hydrology` directory, they can be installed as editable packages using:
```console
uv pip install -e ../neuralhydrology
uv pip install -e ../unite_toolbox
```
Both packages should fulfill the required version in the `pyproject.toml`.

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
