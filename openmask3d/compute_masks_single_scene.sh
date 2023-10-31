#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

MASK_MODULE_CKPT_PATH="/home/ayca/ovseg3d/pretrained_models_and_data/pretrained/scannet200_model.ckpt"
SCENE_PLY_PATH="/home/ayca/openmask3d_temp_clean_up/scene0011_00_vh_clean_2.ply"
MASK_SAVE_DIR="/home/ayca/openmask3d_temp_clean_up/mask_save_dir_scannet200_other_dir"
SAVE_VISUALIZATIONS=false #if set to true, saves pyviz3d visualizations


# TEST
python class_agnostic_mask_computation/get_masks_single_scene.py \
general.experiment_name="single_scene" \
general.checkpoint=${MASK_MODULE_CKPT_PATH} \
general.train_mode=false \
data.test_mode=test \
model.num_queries=150 \
general.use_dbscan=true \
general.dbscan_eps=0.95 \
general.save_visualizations=${SAVE_VISUALIZATIONS} \
general.scene_path=${SCENE_PLY_PATH} \
general.mask_save_dir=${MASK_SAVE_DIR}
