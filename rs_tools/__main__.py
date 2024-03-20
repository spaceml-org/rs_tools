import hydra
from loguru import logger
from omegaconf import OmegaConf, DictConfig


@hydra.main(config_path="../config", config_name="main", version_base="1.3")
def main(cfg):
    lib = hydra.utils.instantiate(cfg...)

    logger.debug(f"Choosing stage...")
    if cfg.stage == "download":
        for satellite in cfg.download.keys():
            # instantiate download function
            # pick parameters
            # download
        logger.debug(f"starting download script...")
        raise NotImplementedError()
    elif cfg.stage == "preprocess":
        logger.debug(f"starting preprocessing script...")
        raise NotImplementedError()
    # elif cfg.stage == "train":
    #     logger.debug(f"starting training script...")
    #     raise NotImplementedError()
    # elif cfg.stage == "inference":
    #     logger.debug(f"starting inference script...")
    #     raise NotImplementedError()
    # elif cfg.stage == "evaluation":
    #     logger.debug(f"starting evaluation script...")
    #     raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized stage: {cfg.stage}")



def main(config: DictConfig):
    pass

if __name__ == "__main__":
    main()

