# Installation


## Conda (Recommended)

We recommend the user to use `conda` with the associated environment for the environment.

```bash
conda env create -f environments/environment.yaml
conda activate rs_tools
```

***
## Pip (Alpha-Version)

We can install via the github repo through pip.

```bash
pip install git+https://github.com/space-ml/rs_tools.git
```

**Warning**: This is an alpha version.

***
## Development Version



```bash
git clone https://github.com/space-ml/rs_tools.git
cd rs_tools
poetry install
```

!!! tip 
    We advise you to create a virtual environment before installing:

    ```bash
    conda env create -f environment.yaml
    conda activate rs_tools
    ```

    and recommend you check your installation passes the supplied unit tests:

    ```bash
    poetry run pytest tests/
    ```

***
## Instrument To Instrument (Work-in-Progress)

We have an example where we could do inference using a pre-trainined model from the ITI repo.
This would require us to install the `itipy` repo directly.

We can use


```bash
conda env create -f environments/environment_iti.yaml
conda activate rs_tools
```


Please see the [iti](https://github.com/spaceml-org/InstrumentToInstrument/tree/development-eo) repo with the [example](https://github.com/spaceml-org/InstrumentToInstrument/blob/development-eo/iti/train/msg_to_goes.py) for more details.


