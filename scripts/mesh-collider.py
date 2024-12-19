import numpy as np
import trimesh
from sklearn.cluster import KMeans
from collections import defaultdict
from skimage import measure
from matplotlib import pyplot as plt 
import os
import time

import argparse

parser = argparse.ArgumentParser(description="Run mesh collider with custom parameters.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the 3D model file.")
parser.add_argument("--k", type=int, required=True, help="Number of clusters.")
parser.add_argument("--grid_size", type=float, required=True, help="Grid size for sweeping and voxelization.")
parser.add_argument("--output_folder", type=str, required=True, help="Output folder for processed results.")
args = parser.parse_args()


# Loads and visualizes a mesh
def import_mesh(mesh_location):
    mesh = trimesh.load(mesh_location)
    mesh.show()
    return mesh

# Converts a mesh into a voxel grid
def voxelize(mesh, grid_size=0.02):
    voxelized_mesh = mesh.voxelized(pitch=grid_size)
    #voxelized_mesh.show()
    return voxelized_mesh

# Clusters voxel points into k groups
def get_cluster(voxel, k):
    voxel_center_points = voxel.points 
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(voxel_center_points)

    # --- From ChatGPT 
    labels = kmeans.labels_
    colors = plt.cm.tab10(np.linspace(0, 1, k))[:, :3]
    voxel_colors = colors[labels] 
    # ---

    rgba_colors = (voxel_colors * 255).astype(np.uint8)
    colored_voxels = trimesh.points.PointCloud(voxel_center_points, colors=rgba_colors)
    #colored_voxels.show()

    # Separate clusters into groups
    clustered = defaultdict(list)
    for cluster, voxel_center_point in zip(labels, voxel_center_points):
        clustered[cluster].append(voxel_center_point)

    clustered = {cluster: np.array(points) for cluster, points in clustered.items()}

    # Visualize each cluster
    for cluster, points in clustered.items():
        show_cluster(points)

    return clustered

# Visualize a single cluster
def show_cluster(points):
    voxel_cluster = trimesh.points.PointCloud(points)
    #voxel_cluster.show()


# Save as .obj file
def save_mesh(mesh, filename, output_folder=None):
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        filename = os.path.join(output_folder, filename)
    try:
        mesh.export(filename)
        print(f"Mesh saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving mesh: {e}")


# Sweeping function
def create_swept_volume_with_depth(cross_section, path, depth=0.02):
    vertices = []
    faces = []
    # Create vertices for top and bottom layers along the path
    for i, point in enumerate(path):
        # Create the top layer
        top_layer = cross_section + point
        # Create the bottom layer
        bottom_layer = cross_section + point + np.array([0, 0, depth])
        # Add both layers to vertices
        vertices.extend(top_layer)
        vertices.extend(bottom_layer)
        
        # Connect adjacent layers
        if i > 0:
            prev_top_start = (i - 1) * len(cross_section) * 2
            prev_bottom_start = prev_top_start + len(cross_section)
            curr_top_start = i * len(cross_section) * 2
            curr_bottom_start = curr_top_start + len(cross_section)
            
            for j in range(len(cross_section)):
                top_v0 = prev_top_start + j
                top_v1 = prev_top_start + (j + 1) % len(cross_section)
                top_v2 = curr_top_start + j
                top_v3 = curr_top_start + (j + 1) % len(cross_section)

                bottom_v0 = prev_bottom_start + j
                bottom_v1 = prev_bottom_start + (j + 1) % len(cross_section)
                bottom_v2 = curr_bottom_start + j
                bottom_v3 = curr_bottom_start + (j + 1) % len(cross_section)

                faces.append([top_v0, top_v1, top_v2])
                faces.append([top_v2, top_v1, top_v3])

                faces.append([bottom_v0, bottom_v1, bottom_v2])
                faces.append([bottom_v2, bottom_v1, bottom_v3])

                faces.append([top_v0, bottom_v0, bottom_v1])
                faces.append([top_v0, bottom_v1, top_v1])
                faces.append([top_v2, bottom_v2, bottom_v3])
                faces.append([top_v2, bottom_v3, top_v3])
    
    # Convert to NumPy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create and return the mesh
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def sweep_and_mesh_cluster(cluster_points, grid_size=0.01, axis='x', step_size=0.05):
    # Define a circular cross-section
    num_points = 20
    theta = np.linspace(0, 2 * np.pi, num_points)
    cross_section = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(num_points)))
    cross_section *= grid_size 

    # Define a sweep path
    path = cluster_points

    # Generate swept volume
    swept_mesh = create_swept_volume_with_depth(cross_section, path)

    # Make the mesh double-sided
    reversed_mesh = swept_mesh.copy()
    reversed_mesh.invert()
    double_sided_mesh = trimesh.util.concatenate([swept_mesh, reversed_mesh])

    return double_sided_mesh

# Merges all mesh clusters into one mesh
def merge_meshes(all_cluster_meshes):
    print("Merging all cluster meshes...")
    combined_mesh = trimesh.util.concatenate(all_cluster_meshes)
    return combined_mesh

def main():
    imported_mesh = import_mesh(args.model_path)
    output_folder = args.output_folder
    start_time = time.time()
    voxelized_mesh = voxelize(imported_mesh)
    clustered = get_cluster(voxelized_mesh, k=args.k)

    # Store initial cluster centroids for alignment
    cluster_centroids = {cluster_id: cluster_points.mean(axis=0) for cluster_id, cluster_points in clustered.items()}

    # Collect all processed cluster meshes
    all_cluster_meshes = []
    for cluster_id, cluster_points in clustered.items():

        # Generate a mesh using sweeping from the cluster points on an axis
        cluster_mesh = sweep_and_mesh_cluster(cluster_points, grid_size=args.grid_size, axis='x', step_size=0.01)
        
        # Compute the offset and apply the translation so clusters align 
        original_centroid = cluster_centroids[cluster_id]
        current_centroid = cluster_mesh.centroid
        offset = original_centroid - current_centroid
        cluster_mesh.apply_translation(offset)

        # Save the cluster mesh used for unity demo. 
        #toggle to True if needed to turn on. 
        toggle = True
        if toggle:
            cluster_filename = os.path.join(output_folder, f"object_{cluster_id}.obj")
            save_mesh(cluster_mesh, filename=cluster_filename)
        #cluster_mesh.show()

        # Add the translated mesh to the list
        all_cluster_meshes.append(cluster_mesh)

    # Combine all cluster meshes into one
    final_combined_mesh = merge_meshes(all_cluster_meshes)
    elapsed_time = time.time() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

    final_filename = "final_object.obj"
    save_mesh(final_combined_mesh, filename=final_filename, output_folder=output_folder)

    try:
        final_combined_mesh.show()
        print("")
    except Exception as e:
        print(f"Failed to visualize the final combined mesh: {e}")


if __name__ == "__main__":
    main()


"""
Script: mesh_processing.py
Description: This script processes 3D meshes by clustering voxelized points, creating swept volumes, and saving clustered and combined meshes.
Sources:
- Trimesh library: https://github.com/mikedh/trimesh
- NumPy library: https://numpy.org/
- scikit-learn (KMeans clustering): https://scikit-learn.org/
- Matplotlib (visualization): https://matplotlib.org/

AI Assistance:
- Portions of this code were generated and refined using OpenAI's ChatGPT, specifically save_mesh and part of get_cluster

Contributors:
- [Joshua Fantillo, John Ordoyo, Silas Myrrh]: Customized, debugged, and optimized the script for specific use cases.
"""
