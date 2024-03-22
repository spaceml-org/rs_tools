import hydra
from typing import Dict
from loguru import logger
from omegaconf import OmegaConf, DictConfig


@hydra.main(config_path="../config/example/", config_name="main", version_base="1.3")
def main(cfg):
    # instantiate the config 
    logger.debug(f"Choosing Stage...")
    if cfg.stage == "download":
        logger.info(f"Instantiating Datamodule: {cfg.satellite.download['_target_']}")
        download = hydra.utils.instantiate(cfg.satellite.download)
        logger.info(f"Starting Download")
        hydra.utils.call(download)

    elif cfg.stage == "geoprocess":
        logger.debug(f"starting geoprocessing script...")
        raise NotImplementedError()
    
    elif cfg.stage == "preprocess":
        logger.debug(f"starting preprocessing script...")
        raise NotImplementedError()
    
    elif cfg.stage == "patch":
        logger.debug(f"starting patching script...")
        raise NotImplementedError()
    
    else:
        raise ValueError(f"Unrecognized stage: {cfg.stage}")

if __name__ == "__main__":
    main()

""""
EXAMPLES:
python rs_tools satellite=goes
"""

