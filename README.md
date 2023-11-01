<p align="center">

  <h1 align="center">OpenMask3D🛋: Open-Vocabulary 3D Instance Segmentation</h1>
  <p align="center">
    <a href="https://aycatakmaz.github.io/">Ay&#231;a Takmaz</a><sup>1*</sup></span>, 
    <a href="https://elisabettafedele.github.io/">Elisabetta Fedele</a><sup>1*</sup>
     <br>
    <a href="https://studios.disneyresearch.com/people/bob-sumner/">Robert W. Sumner</a><sup>1</sup>, 
    <a href="https://people.inf.ethz.ch/pomarc/">Marc Pollefeys</a><sup>1,2</sup>, 
    <a href="https://federicotombari.github.io/">Federico Tombari</a><sup>1,3</sup>,
    <a href="https://francisengelmann.github.io/">Francis Engelmann</a><sup>1,3</sup>
    <br>
    <sup>1</sup>ETH Zurich, 
    <sup>2</sup>Microsoft, 
    <sup>3</sup>Google <br>
    <sup>*</sup>equal contribution
  </p>
  <h2 align="center">NeurIPS 2023</h2>
  <h3 align="center"><a href="https://github.com/OpenMask3D/openmask3d">Code</a> | <a href="https://arxiv.org/abs/2306.13631">Paper</a> | <a href="https://openmask3d.github.io">Project Page</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://openmask3d.github.io/static/images/teaser.jpeg" alt="Logo" width="100%">
  </a>
</p>
<p align="center">
<strong>OpenMask3D</strong> is a zero-shot approach for 3D instance segmentation with open-vocabulary queries.
Guided by predicted class-agnostic 3D instance masks, our model aggregates per-mask features via multi-view fusion of CLIP-based image embeddings.
</p>
<br>

---
## Setup 🛠
Clone the repository, create conda environment and install the required packages as follows:
```bash
conda create --name=openmask3d python=3.8.5 # create new virtual environment
conda activate openmask3d # activate it
bash install_requirements.sh  # install requirements
pip install -e .  # install current repository in editable mode
```
Note: If you encounter any issues in the `bash install_requirements.sh` step, we recommend you to run the commands in that script one-by-one, especially for performing the MinkowskiEngine installation manually. 

---

## Run the pipeline on a single scene 🛋
In this section we provide some information about how to run the pipeline on a single scene. In particular, we divide this section into four parts:
1. Download **checkpoints**
2. Check the format of **scene's data**
3. Set-up **configurations** 
4. **Run** OpenMask3D 

### Step 1: Download the checkpoints 📍
Create a folder `resources` in the main directory of the repository. Then, add to this folder the checkpoints for:
* **Mask module network**: use [this link](https://drive.google.com/file/d/1emtZ9xCiCuXtkcGO3iIzIRzcmZAFfI_B/view?usp=sharing) (model trained on ScanNet200 training set) for evaluating on **ScanNet validation scenes**, or [this link](https://drive.google.com/file/d/1rD2Uvbsi89X4lSkont_jUTT7X9iaox9y/view?usp=share_link) for running the model on an **arbitrary scene**.
* **Segment Anything Model** (in our case we used ViT-H): use this [link](https://drive.google.com/file/d/1WHi0hBi0iqMZfk8l3rDXLrW4lEEgHm_y/view?usp=sharing) or the [official repository](https://github.com/facebookresearch/segment-anything#model-checkpoints).

### Step 2: Check the folder structure of the data for your scene 🛢
In order to run OpenMask3D you need to have access to the point cloud of the scene as well to the posed RGB-D frames.

We recommend creating a folder `scene_example` inside the `resources` folder where the data is saved with the following structure ([here](https://drive.google.com/file/d/1UOwBZMCrTMg-_MFwmYkKOrex1YS6Nw-i/view?usp=sharing) we provide a scene as an example). 
```
scene_example
      ├── pose                            <- folder with camera poses
      │      ├── 0.txt 
      │      ├── 1.txt 
      │      └── ...  
      ├── color                           <- folder with RGB images
      │      ├── 0.jpg (or .png/.jpeg)
      │      ├── 1.jpg (or .png/.jpeg)
      │      └── ...  
      ├── depth                           <- folder with depth images
      │      ├── 0.png (or .jpg/.jpeg)
      │      ├── 1.png (or .jpg/.jpeg)
      │      └── ...  
      ├── intrinsic                 
      │      └── intrinsic_color.txt       <- camera intrinsics
      └── scene_example.ply                <- point cloud of the scene
```

Please note the followings:
* The **point cloud** should be provided as a `.ply` file and the points are expected to be in the z-up right-handed coordinate system.
* The **camera intrinsics** and **camera poses** should be provided in a `.txt` file, containing a 4x4 matrix.
* The **RGB images** and the **depths** can be either in `.png`, `.jpg`, `.jpeg` format; the used format should be specified as explained in **Step 3**.
* The **RGB images** and their corresponding **depths** and **camera poses** should be named as `{FRAME_ID}.extension`, without zero padding for the frame ID, starting from index 0.

### Step 3: Set-up the paths to data and to output folders 🛤
Before running OpenMask3D make sure to fill all the required parameters in [this script](run_openmask3d_single_scene.sh). In particular, if you have followed the structure provided in Step 2, you should adapt only the following fields:
* `SCENE_DIR`: directory to `scene_example`
* `SCENE_INTRINSIC_RESOLUTION`: resolution on which intrinsics are computed
* `IMG_EXTENSION`: extension of RGB pictures. Either `.png`, `.jpg`, `.jpeg`
* `DEPTH_EXTENSION`: extension of depth pictures. Either `.png`, `.jpg`, `.jpeg`
* `DEPTH_SCALE`: factor by which the depth of the sensor should be divided to obtain a measure in terms of meters. It should be set to 1000 for ScanNet depth images and to 6553.5 for Replica depth images. You should set this value based on the scale of your depth maps.
* `MASK_MODULE_CKPT_PATH`: path to the mask module network checkpoint
* `SAM_CKPT_PATH`: path to the Segment Anything Model (SAM) checkpoint
* `OUTPUT_FOLDER_DIRECTORY`: path to the folder in which you wish to save the outputs
* `SAVE_VISUALIZATIONS`: set to true if you wish to save the visualizations of the class-agnostic masks
* `SAVE_CROPS`: set to true if you wish to save the 2D crops of the masks from which the CLIP features are extracted. It can be helpful for debugging and for a qualitative evaluation of the quality of the masks.
* `OPTIMIZE_GPU_USAGE`: set to true if you have some memory constraints and wish to minimize GPU memory footprint. Please note that this version is slower compared to the our default version.


### Step 4: Run OpenMask3D 🚀
Now you can run OpenMask3D by using the following command.
```bash
bash run_openmask3d_single_scene.sh
```
This script first extracts and saves the class-agnostic masks, and then computes the per-mask features. Masks and mask-features are saved into the directory specified by the user at the beginning of [this script](run_openmask3d_single_scene.sh). In particular, the output has the following structure.
```
OUTPUT_FOLDER_DIRECTORY
      └── date-time-experiment_name                           <- folder with the output of a specific experiment
             ├── crops                                        <- folder with crops (if SAVE_CROPS=true)
             ├── hydra_outputs                                <- folder with outputs from hydra (config.yaml files are useful)
             ├── scene_example_masks.pt                       <- class-agnostic instance masks - dim. (num_points, num_masks) indicating the masks in which a given point is included
             └── scene_example_openmask3d_features.npy        <- per-mask features for each object instance - dim. (num_masks, num_features), the mask-feature vecture for each instance mask. 
```


Note: For the ScanNet validation, we use available segments on ScanNet and obtain more robust and less noisy masks compared to directly running the mask predictor on the point cloud. Therefore, the results we obtain for a single scene from ScanNet directly using the point cloud can be different then the masks obtained during the overall ScanNet evaluation described in the section below. 

---
## Other Configs ⚙️
Other configuration parameters can be modified from [this file](openmask3d/configs/openmask3d_inference.yaml). Here we provide some clarifications of other configuration parameters:
- `multi_level_expansion_ratio`: factor of increment of the crop dimension for using multi-level image crops
- `openmask3d.frequency`: the frequency with which we want to process the frames given in input (e.g. a frequency of 10 takes 1 image in every 10 frames)
- `openmask3d.num_random_rounds` and `openmask3d.num_selected_points`: sets the number of iterations and the number of sampled points for SAM.

---
## Closed-vocabulary 3D instance segmentation evaluation on ScanNet200 📊
In this section we outline the steps to take in order to reproduce our results on the ScanNet200 validation set. In particular, we divide this section into four parts:
1. Download and preprocess the **ScanNet200** dataset
2. Check the format of ScanNet200 dataset
3. Set-up the paths to data and to output folders
2. Run evaluation


### Step 1: Download and pre-process the ScanNet200 dataset 📍
First, you need to download the ScanNet200 dataset as explained [here](https://kaldir.vc.in.tum.de/scannet_benchmark/documentation).

Once you have the dataset, you have to clone the [ScanNet repository](https://github.com/ScanNet/ScanNet) and process the dataset by using the following command.
```
cd class_agnostic_mask_computation
python -m datasets.preprocessing.scannet_preprocessing preprocess \
--data_dir="PATH_TO_ORIGINAL_SCANNET_DATASET" \
--save_dir="data/processed/scannet" \
--git_repo="PATH_TO_SCANNET_GIT_REPO" \
--scannet200=true
```
### Step 2: Check the format of ScanNet200 dataset 🛢
Make sure to have the data in the following form.
```bash
scans                                       <- out folder          
 ├── scene_0011_00
 │     ├── data 
 │     │      ├── intrinsic                 <- folder with the intrinsics
 │     │      └── pose                      <- folder with the poses
 │     ├── data_compressed                 
 │     │      ├── color                     <- folder with the color images
 │     │      └── depth                     <- folder with the depth images
 │     └── scene_0011_00_vh_clean_2.ply     <- path to the point cloud/mesh ply file
 ├── scene0011_01
 │     ├── data 
 │     │      ├── intrinsic
 │     │      └── pose     
 │     ├── data_compressed                 
 │     │      ├── color
 │     │      └── depth  
 │     └── scene_0011_01_vh_clean_2.ply 
 ...
```

### Step 3: Set-up paths to data and to output folders 🛤
Modify the paths and parameters in [this script](run_openmask3d_scannet200_eval.sh), following the instructions provided there.

### Step 4: Run OpenMask3D on ScanNet200 🚀
 Now you can compute the per-mask scene features and run the evaluation of OpenMask3D on the whole ScanNet200 dataset by using the following command:
 ```bash
bash run_openmask3d_scannet200_eval.sh
```
This script first extracts and saves the class-agnostic masks, and then computes the mask features associated with each extracted mask. Afterwards, the evaluation script automatically runs in order to obtain 3D closed-vocabulary semantic instance segmentation scores. 

---
## Citation :pray:
```
@inproceedings{takmaz2023openmask3d,
  title={{OpenMask3D: Open-Vocabulary 3D Instance Segmentation}},
  author={Takmaz, Ay{\c{c}}a and Fedele, Elisabetta and Sumner, Robert W. and Pollefeys, Marc and Tombari, Federico and Engelmann, Francis},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
year={2023}
}
```
