import numpy as np
import open3d as o3d
import torch
import clip
import pdb
import matplotlib.pyplot as plt
from constants import *


class QuerySimilarityComputation():
    def __init__(self,):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.clip_model, _ = clip.load('ViT-L/14@336px', self.device)

    def get_query_embedding(self, text_query):
        text_input_processed = clip.tokenize(text_query).to(self.device)
        with torch.no_grad():
            sentence_embedding = self.clip_model.encode_text(text_input_processed)

        sentence_embedding_normalized =  (sentence_embedding/sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
        return sentence_embedding_normalized.squeeze().numpy()
 
    def compute_similarity_scores(self, mask_features, text_query):
        text_emb = self.get_query_embedding(text_query)

        scores = np.zeros(len(mask_features))
        for mask_idx, mask_emb in enumerate(mask_features):
            mask_norm = np.linalg.norm(mask_emb)
            if mask_norm < 0.001:
                continue
            normalized_emb = (mask_emb/mask_norm)
            scores[mask_idx] = normalized_emb@text_emb

        return scores
    
    def get_per_point_colors_for_similarity(self, 
                                            per_mask_scores, 
                                            masks, 
                                            normalize_based_on_current_min_max=False, 
                                            normalize_min_bound=0.16, #only used for visualization if normalize_based_on_current_min_max is False
                                            normalize_max_bound=0.26, #only used for visualization if normalize_based_on_current_min_max is False
                                            background_color=(0.77, 0.77, 0.77)
                                        ):
        # get colors based on the openmask3d per mask scores
        non_zero_points = per_mask_scores!=0
        openmask3d_per_mask_scores_rescaled = np.zeros_like(per_mask_scores)
        pms = per_mask_scores[non_zero_points]

        # in order to be able to visualize the score differences better, we can use a normalization scheme
        if normalize_based_on_current_min_max: # if true, normalize the scores based on the min. and max. scores for this scene
            openmask3d_per_mask_scores_rescaled[non_zero_points] = (pms-pms.min()) / (pms.max() - pms.min())
        else: # if false, normalize the scores based on a pre-defined color scheme with min and max clipping bounds, normalize_min_bound and normalize_max_bound.
            new_scores = np.zeros_like(openmask3d_per_mask_scores_rescaled)
            new_indices = np.zeros_like(non_zero_points)
            new_indices[non_zero_points] += pms>normalize_min_bound
            new_scores[new_indices] = ((pms[pms>normalize_min_bound]-normalize_min_bound)/(normalize_max_bound-normalize_min_bound))
            openmask3d_per_mask_scores_rescaled = new_scores

        new_colors = np.ones((masks.shape[1], 3))*0 + background_color
        
        for mask_idx, mask in enumerate(masks[::-1, :]):
            # get color from matplotlib colormap
            new_colors[mask>0.5, :] = plt.cm.jet(openmask3d_per_mask_scores_rescaled[len(masks)-mask_idx-1])[:3]

        return new_colors



def main():
    # --------------------------------
    # Set the paths
    # --------------------------------
    path_scene_pcd = "data/scene_example.ply"
    path_pred_masks = "data/scene_example_masks.pt"
    path_openmask3d_features = "data/scene_example_openmask3d_features.npy"
    

    # --------------------------------
    # Load data
    # --------------------------------
    # load the scene pcd
    scene_pcd = o3d.io.read_point_cloud(path_scene_pcd)
    
    # load the predicted masks
    pred_masks = np.asarray(torch.load(path_pred_masks)).T # (num_instances, num_points)

    # load the openmask3d features
    openmask3d_features = np.load(path_openmask3d_features) # (num_instances, 768)

    # initialize the query similarity computer
    query_similarity_computer = QuerySimilarityComputation()
    

    # --------------------------------
    # Set the query text
    # --------------------------------
    query_text = "ENTER QUERY TEXT HERE" # change the query text here


    # --------------------------------
    # Get the similarity scores
    # --------------------------------
    # get the per mask similarity scores, i.e. the cosine similarity between the query embedding and each openmask3d mask-feature for each object instance
    per_mask_query_sim_scores = query_similarity_computer.compute_similarity_scores(openmask3d_features, query_text)


    # --------------------------------
    # Visualize the similarity scores
    # --------------------------------
    # get the per-point heatmap colors for the similarity scores
    per_point_similarity_colors = query_similarity_computer.get_per_point_colors_for_similarity(per_mask_query_sim_scores, pred_masks) # note: for normalizing the similarity heatmap colors for better clarity, you can check the arguments for the function get_per_point_colors_for_similarity

    # visualize the scene with the similarity heatmap
    scene_pcd_w_sim_colors = o3d.geometry.PointCloud()
    scene_pcd_w_sim_colors.points = scene_pcd.points
    scene_pcd_w_sim_colors.colors = o3d.utility.Vector3dVector(per_point_similarity_colors)
    scene_pcd_w_sim_colors.estimate_normals()
    o3d.visualization.draw_geometries([scene_pcd_w_sim_colors])
    # alternatively, you can save the scene_pcd_w_sim_colors as a .ply file
    # o3d.io.write_point_cloud("data/scene_pcd_w_sim_colors_{}.ply".format('_'.join(query_text.split(' '))), scene_pcd_w_sim_colors)

if __name__ == "__main__":
    main()
