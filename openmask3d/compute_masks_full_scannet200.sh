#!/bin/bash
export OMP_NUM_THREADS=4  # speeds up MinkowskiEngine

MASK_MODULE_CKPT_PATH="/home/ayca/ovseg3d/pretrained_models_and_data/pretrained/scannet200_val.ckpt"
SCANNET_DATA_DIR="/media/ayca/Elements/ayca/OpenMask3D/scannet_processed/scannet200"
SCANNET_LABEL_DB_PATH="${SCANNET_DATA_DIR%/}/label_database.yaml"
MASK_SAVE_DIR="/home/ayca/openmask3d_temp_clean_up/mask_save_dir_scannet200_other_dir"
SAVE_VISUALIZATIONS=false #if set to true, saves pyviz3d visualizations

# TEST
python class_agnostic_mask_computation/get_masks_scannet200.py \
general.experiment_name="scannet200" \
general.project_name="scannet200" \
general.checkpoint=${MASK_MODULE_CKPT_PATH} \
general.train_mode=false \
model.num_queries=150 \
general.use_dbscan=true \
general.dbscan_eps=0.95 \
general.save_visualizations=${SAVE_VISUALIZATIONS} \
data.test_dataset.data_dir=${SCANNET_DATA_DIR}  \
data.validation_dataset.data_dir=${SCANNET_DATA_DIR} \
data.train_dataset.data_dir=${SCANNET_DATA_DIR} \
data.test_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH}  \
data.validation_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH} \
data.train_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH} \
general.mask_save_dir=${MASK_SAVE_DIR}
