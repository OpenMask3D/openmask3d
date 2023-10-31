set -e

# OpenMask3D Installation
#   - If you encounter any problem with the Detectron2 or MinkowskiEngine installations, 
#     it might be because you don't have properly set up gcc, g++, pybind11, openblas installations.
#     First, make sure you have those are installed properly. 
#   - More details about installing on different platforms can be found in the GitHub repositories of 
#     Detectron2(https://github.com/facebookresearch/detectron2) and MinkowskiEngine (https://github.com/NVIDIA/MinkowskiEngine).
#   - If you encounter any other problems, take a look at the installation guidelines in https://github.com/JonasSchult/Mask3D, which might be helpful as our mask module relies on Mask3D.

# Note: The following commands were tested on Ubuntu 18.04 and 20.04, with CUDA 11.1 and 11.4.

pip install torch==1.12.1 torchvision==0.13.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install ninja==1.10.2.3
pip install pytorch-lightning==1.7.2 fire==0.5.0 imageio==2.23.0 tqdm==4.64.1 wandb==0.13.2
pip install python-dotenv==0.21.0 pyviz3d==0.2.32 scipy==1.9.3 plyfile==0.7.4 scikit-learn==1.2.0 trimesh==3.17.1 loguru==0.6.0 albumentations==1.3.0 volumentations==0.1.8
pip install antlr4-python3-runtime==4.8 black==21.4b2 omegaconf==2.0.6 hydra-core==1.0.5 --no-deps

pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

conda install -y openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --config-settings="--blas_include_dirs=${CONDA_PREFIX}/include" --config-settings="--blas=openblas" 

pip install pynvml==11.4.1 gpustat==1.0.0 tabulate==0.9.0 pytest==7.2.0 tensorboardx==2.5.1 yapf==0.32.0 termcolor==2.1.1 addict==2.4.0 blessed==1.19.1
pip install gorilla-core==0.2.7.8
pip install matplotlib==3.7.2
pip install cython 

pip install pycocotools==2.0.6
pip install h5py==3.7.0
pip install transforms3d==0.4.1
pip install open3d==0.16.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torchmetrics==0.11.0
pip install setuptools==68.0.0

pip install fvcore==0.1.5.post20221221 
pip install cloudpickle==2.1.0
pip install Pillow==9.3.0

cd openmask3d/class_agnostic_mask_computation/third_party/pointnet2 && pip install .

pip install git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16 --no-deps
pip install  git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588 --no-deps
pip install ftfy==6.1.1
pip install regex==2023.10.3