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


def load_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    pcd.estimate_normals()
    coords = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    return coords, colors, normals

def process_file(filepath):
    coords, colors, normals = load_ply(filepath)
    raw_coordinates = coords.copy()
    raw_colors = (colors*255).astype(np.uint8)
    raw_normals = normals

    features = colors
    if len(features.shape) == 1:
        features = np.hstack((features[None, ...], coords))
    else:
        features = np.hstack((features, coords))

    filename = filepath.split("/")[-1][:-4]
    return [[coords, features, [], filename, raw_colors, raw_normals, raw_coordinates, 0]] # 2: original_labels, 3: none
    # coordinates, features, labels, self.data[idx]['raw_filepath'].split("/")[-2], raw_color, raw_normals, raw_coordinates, idx

@hydra.main(config_path="conf", config_name="config_base_class_agn_masks_single_scene.yaml")
def get_class_agnostic_masks(cfg: DictConfig):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)

    c_fn = hydra.utils.instantiate(cfg.data.test_collation) #(model.config.data.test_collation)

    input_batch = process_file(cfg.general.scene_path)
    batch = c_fn(input_batch)

    model.to(device)
    model.eval()

    start = time.time()
    with torch.no_grad():
        res_dict = model.get_masks_single_scene(batch)
    end = time.time()
    print("Time elapsed: ", end - start)

@hydra.main(config_path="conf", config_name="config_base_class_agn_masks_single_scene.yaml")
def main(cfg: DictConfig):
    get_class_agnostic_masks(cfg)

if __name__ == "__main__":
    main()
