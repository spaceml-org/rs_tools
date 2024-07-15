""""
EXAMPLES:
    python rs_tools satellite=goes stage=download
    python rs_tools satellite=goes stage=geoprocess
    python rs_tools satellite=msg stage=geoprocess
    python rs_tools satellite=terra stage=geoprocess
"""


import hydra
from typing import Dict
from loguru import logger
from omegaconf import OmegaConf, DictConfig
import autoroot

# TODO add option to provide config path via command line

@hydra.main(config_path="../config/walkthrough/", config_name="main", version_base="1.3")
def main(cfg):
    # instantiate the config 
    logger.debug(f"Choosing Stage...")
    if cfg.stage == "query":
        logger.info(f"Instantiating Downloader: {cfg.satellite.query['_target_']}")
        query = hydra.utils.instantiate(cfg.satellite.query)
    # instantiate the config 
    elif cfg.stage == "download":
        logger.info(f"Instantiating Downloader: {cfg.satellite.download['_target_']}")
        download = hydra.utils.instantiate(cfg.satellite.download)
        # hydra.utils.call(download)

    elif cfg.stage == "geoprocess":
        logger.info(f"Instantiating Geoprocessor: {cfg.satellite.geoprocess['_target_']}")
        geoprocess = hydra.utils.instantiate(cfg.satellite.geoprocess)
        print(geoprocess)
        hydra.utils.call(geoprocess)
    
    elif cfg.stage == "patch":
        logger.debug(f"Instantiating Prepatcher: {cfg.satellite.patch['_target_']}")
        patch = hydra.utils.instantiate(cfg.satellite.patch)
        hydra.utils.call(patch)
    else:
        raise ValueError(f"Unrecognized stage: {cfg.stage}")

if __name__ == "__main__":
    root_dir = autoroot.root
    main()

