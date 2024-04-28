# Installation


We can install via the github repo

```bash
pip install git+https://github.com/space-ml/rs_tools.git
```

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



