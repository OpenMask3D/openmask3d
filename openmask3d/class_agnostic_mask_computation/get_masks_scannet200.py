import logging
import os
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys
)
from pytorch_lightning import Trainer
import open3d as o3d
import numpy as np
import torch
import time
import pdb

def get_parameters(cfg: DictConfig):
    #logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    #loggers = []

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    #logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, None #loggers


@hydra.main(config_path="conf", config_name="config_base_class_agn_masks_scannet200.yaml")
def get_class_agnostic_masks_scannet200(cfg: DictConfig):
   
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, _ = get_parameters(cfg)
    test_dataset = hydra.utils.instantiate(cfg.data.test_dataset)
    c_fn = hydra.utils.instantiate(cfg.data.test_collation)

    test_dataloader = hydra.utils.instantiate(
            cfg.data.test_dataloader,
            test_dataset,
            collate_fn=c_fn,
        )
    model.freeze()
    #print(list(test_dataloader))
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=None,
        **cfg.trainer
    )
    runner.test(model)


@hydra.main(config_path="conf", config_name="config_base_class_agn_masks_scannet200.yaml")
def main(cfg: DictConfig):
    get_class_agnostic_masks_scannet200(cfg)

if __name__ == "__main__":
    main()
