import hydra
from omegaconf import DictConfig
import numpy as np
from openmask3d.data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from openmask3d.utils import get_free_gpu, create_out_folder
from openmask3d.mask_features_computation.features_extractor import FeaturesExtractor
import torch
import os
from glob import glob

# TIP: add version_base=None to the arguments if you encounter some error  
@hydra.main(config_path="configs", config_name="openmask3d_scannet200_eval")
def main(ctx: DictConfig):
    device = "cpu"  # "mps" if torch.backends.mps.is_available() else "cpu"
    device = get_free_gpu(7000) if torch.cuda.is_available() else device
    print(f"[INFO] Using device: {device}")
    out_folder = ctx.output.output_directory
    os.chdir(hydra.utils.get_original_cwd())
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print(f"[INFO] Saving feature results to {out_folder}")
    masks_paths = sorted(glob(os.path.join(ctx.data.masks.masks_path, ctx.data.masks.masks_suffix)))
    
    for masks_path in masks_paths:
        
        scene_num_str = masks_path.split('/')[-1][5:12]
        path = os.path.join(ctx.data.scans_path, 'scene'+ scene_num_str)
        poses_path = os.path.join(path,ctx.data.camera.poses_path)
        point_cloud_path = glob(os.path.join(path, '*vh_clean_2.ply'))[0]
        intrinsic_path = os.path.join(path, ctx.data.camera.intrinsic_path)
        images_path = os.path.join(path, ctx.data.images.images_path)
        depths_path = os.path.join(path, ctx.data.depths.depths_path)
        
        # 1. Load the masks
        masks = InstanceMasks3D(masks_path) 

        # 2. Load the images
        indices = np.arange(0, get_number_of_images(poses_path), step = ctx.openmask3d.frequency)
        images = Images(images_path=images_path, 
                        extension=ctx.data.images.images_ext, 
                        indices=indices)

        # 3. Load the pointcloud
        pointcloud = PointCloud(point_cloud_path)

        # 4. Load the camera configurations
        camera = Camera(intrinsic_path=intrinsic_path, 
                        intrinsic_resolution=ctx.data.camera.intrinsic_resolution, 
                        poses_path=poses_path, 
                        depths_path=depths_path, 
                        extension_depth=ctx.data.depths.depths_ext, 
                        depth_scale=ctx.data.depths.depth_scale)

        # 5. Run extractor
        features_extractor = FeaturesExtractor(camera=camera, 
                                                clip_model=ctx.external.clip_model, 
                                                images=images, 
                                                masks=masks,
                                                pointcloud=pointcloud, 
                                                sam_model_type=ctx.external.sam_model_type,
                                                sam_checkpoint=ctx.external.sam_checkpoint,
                                                vis_threshold=ctx.openmask3d.vis_threshold,
                                                device=device)

        features = features_extractor.extract_features(topk=ctx.openmask3d.top_k, 
                                                        multi_level_expansion_ratio = ctx.openmask3d.multi_level_expansion_ratio,
                                                        num_levels=ctx.openmask3d.num_of_levels, 
                                                        num_random_rounds=ctx.openmask3d.num_random_rounds,
                                                        num_selected_points=ctx.openmask3d.num_selected_points,
                                                        save_crops=ctx.output.save_crops,
                                                        out_folder=out_folder,
                                                        optimize_gpu_usage=ctx.gpu.optimize_gpu_usage)
        
        # 6. Save features
        filename = f"scene{scene_num_str}_openmask3d_features.npy"
        output_path = os.path.join(out_folder, filename)
        np.save(output_path, features)
        print(f"[INFO] Mask features for scene {scene_num_str} saved to {output_path}.")
    
    
    
if __name__ == "__main__":
    main()