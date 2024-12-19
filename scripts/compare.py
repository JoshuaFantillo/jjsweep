from scipy.spatial import cKDTree
import numpy as np
import trimesh
import time


# Get chamfer distance
def chamfer_distance(points1, points2):
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    dist1, _ = tree1.query(points2, k=1)  
    dist2, _ = tree2.query(points1, k=1)  

    chamfer = np.mean(dist1**2) + np.mean(dist2**2)
    return chamfer

# Get hausdorff distance
def hausdorff_distance(points1, points2):
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    dist1, _ = tree1.query(points2, k=1)
    dist2, _ = tree2.query(points1, k=1)

    hausdorff = max(np.max(dist1), np.max(dist2))
    return hausdorff

# Get MSE
def voxel_mse(original_voxel, reconstructed_voxel):
    original_voxel, reconstructed_voxel = align_voxel_shapes(original_voxel, reconstructed_voxel)

    difference = np.logical_xor(original_voxel, reconstructed_voxel)

    return np.mean(difference.astype(float))

# Align 2 voxels
def align_voxel_shapes(voxel1, voxel2):
    shape1 = np.array(voxel1.shape)
    shape2 = np.array(voxel2.shape)

    max_shape = np.maximum(shape1, shape2)

    pad1 = [(0, max_dim - dim) for dim, max_dim in zip(shape1, max_shape)]
    aligned_voxel1 = np.pad(voxel1, pad1, mode='constant', constant_values=0)

    pad2 = [(0, max_dim - dim) for dim, max_dim in zip(shape2, max_shape)]
    aligned_voxel2 = np.pad(voxel2, pad2, mode='constant', constant_values=0)

    return aligned_voxel1, aligned_voxel2


def timed_step(step_description, func, *args, **kwargs):
    start_time = time.time()
    print(f"{step_description}...")
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    print(f"{step_description} completed in {elapsed_time:.2f} seconds.")
    return result

def main():
    # Step 1: Load models
    print("getting model cat")
    original_model_path = "data/cat/cat_model.obj"
    print("getting final model")
    reconstructed_model_path = "output/final_object.obj"

    print("getting original mesh")
    original_mesh = timed_step("[1/5] Loading original model", trimesh.load, original_model_path)
    print("getting reconstructed mesh")
    reconstructed_mesh = timed_step("[1/5] Loading reconstructed model", trimesh.load, reconstructed_model_path)

    # Step 2: Sample points
    original_points = timed_step("[2/5] Sampling points from original model", original_mesh.sample, 1000)
    reconstructed_points = timed_step("[2/5] Sampling points from reconstructed model", reconstructed_mesh.sample, 1000)

    # Step 3: Chamfer Distance
    chamfer = timed_step("[3/5] Computing Chamfer Distance", chamfer_distance, original_points, reconstructed_points)
    print("Chamfer Distance:", chamfer)

    # Step 4: Hausdorff Distance
    hausdorff = timed_step("[4/5] Computing Hausdorff Distance", hausdorff_distance, original_points, reconstructed_points)
    print("Hausdorff Distance:", hausdorff)

    # Step 5: Voxelize and compute MSE
    voxel_size = 0.1
    original_voxel = timed_step("[5/5] Voxelizing original model", original_mesh.voxelized, voxel_size).matrix
    reconstructed_voxel = timed_step("[5/5] Voxelizing reconstructed model", reconstructed_mesh.voxelized, voxel_size).matrix

    mse = timed_step("[5/5] Computing Voxel MSE", voxel_mse, original_voxel, reconstructed_voxel)
    print("Voxel MSE:", mse)

    print("All computations complete!")


        # Step 1: Load models
    print("getting model cat")
    baseline_model_path = "output/cat_spheres.obj"

    print("getting baseline mesh")
    baseline_mesh = timed_step("[1/5] Loading original model", trimesh.load, baseline_model_path)

    # Step 2: Sample points
    baseline_points = timed_step("[2/5] Sampling points from original model", baseline_mesh.sample, 1000)

    # Step 3: Chamfer Distance
    chamfer = timed_step("[3/5] Computing Chamfer Distance", chamfer_distance, original_points, baseline_points)
    print("Chamfer Baseline Distance:", chamfer)

    # Step 4: Hausdorff Distance
    hausdorff = timed_step("[4/5] Computing Hausdorff Distance", hausdorff_distance, original_points, baseline_points)
    print("Hausdorff Baseline Distance:", hausdorff)

    # Step 5: Voxelize and compute MSE
    voxel_size = 0.1
    original_voxel = timed_step("[5/5] Voxelizing original model", original_mesh.voxelized, voxel_size).matrix
    baseline_voxel = timed_step("[5/5] Voxelizing reconstructed model", baseline_points.voxelized, voxel_size).matrix

    mse = timed_step("[5/5] Computing Voxel MSE", voxel_mse, original_voxel, baseline_voxel)
    print("Voxel Baseline MSE:", mse)

    print("All computations complete!")



if __name__ == "__main__":
    main()

"""
Script: compare.py
Description: This script performs comparison metrics (Chamfer Distance, Hausdorff Distance, and Voxel MSE) between reconstructed 3D models, their original counterparts, and baseline models. It includes functionality for sampling, voxelizing, and aligning shapes.

Sources:
- **SciPy `cKDTree`**: Used for nearest-neighbor queries to compute Chamfer Distance and Hausdorff Distance.
  - Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
- **Trimesh**: Used for loading, sampling, and voxelizing 3D models.
  - URL: https://github.com/mikedh/trimesh
- **NumPy**: Used for numerical operations, including logical operations and array manipulations.
  - Documentation: https://numpy.org/doc/

Contributors:
- Joshua Fantillo, John Ordoyo, Silas Myrrh: Developed the script to streamline 3D model comparison, optimized voxel alignment, and added runtime tracking for each step.

AI Assistance:
- Portions of the code and logic, including modularization of functions and optimizations, were refined with the help of OpenAI's ChatGPT.

Usage:
- Replace the hardcoded paths (`original_model_path`, `reconstructed_model_path`, `baseline_model_path`) with your specific model paths if necessary.
- The script computes metrics for both reconstructed models and baseline models in comparison to the original model.

Citations for Libraries:
1. **SciPy**: Scipy library for scientific computing.
   - URL: https://www.scipy.org/
2. **Trimesh**: A Python library for triangular meshes.
   - URL: https://github.com/mikedh/trimesh
3. **NumPy**: A Python library for large, multi-dimensional arrays and matrices.
   - URL: https://numpy.org/
"""
