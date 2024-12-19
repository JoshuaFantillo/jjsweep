import os
import trimesh
import argparse
from compare import chamfer_distance, hausdorff_distance, voxel_mse

default_config = [
    {
        "name": "cat",
        "output_path": "output/cat/final_object.obj",
        "original_path": "data/cat/cat_model.obj",
        "baseline_path": "output/cat/spheres.obj",
    },
    {
        "name": "dog",
        "output_path": "output/dog/final_object.obj",
        "original_path": "data/dog/dog.obj",
        "baseline_path": "output/dog/spheres.obj",
    },
    {
        "name": "hand",
        "output_path": "output/hand/final_object.obj",
        "original_path": "data/hand/hand.obj",
        "baseline_path": "output/hand/spheres.obj",
    },
    {
        "name": "pot",
        "output_path": "output/pot/final_object.obj",
        "original_path": "data/pot/pot.obj",
        "baseline_path": "output/pot/spheres.obj",
    },
    {
        "name": "rod",
        "output_path": "output/rod/final_object.obj",
        "original_path": "data/rod/rod.obj",
        "baseline_path": "output/rod/spheres.obj",
    },
    {
        "name": "shiba",
        "output_path": "output/shiba/final_object.obj",
        "original_path": "data/shiba/shiba_model.obj",
        "baseline_path": "output/shiba/spheres.obj",
    },
    {
        "name": "sofa",
        "output_path": "output/sofa/final_object.obj",
        "original_path": "data/sofa/sofa.obj",
        "baseline_path": "output/sofa/spheres.obj",
    },
    {
        "name": "starfish",
        "output_path": "output/starfish/final_object.obj",
        "original_path": "data/starfish/starfish_model.obj",
        "baseline_path": "output/starfish/spheres.obj",
    },
]




def compare_models(model_name, output_path, original_path, baseline_path):
    print(f"Comparing models for {model_name}...")

    # Load meshes
    original_mesh = trimesh.load(original_path)
    output_mesh = trimesh.load(output_path)
    baseline_mesh = trimesh.load(baseline_path)

    # Sample points for distances
    original_points = original_mesh.sample(1000)
    output_points = output_mesh.sample(1000)
    baseline_points = baseline_mesh.sample(1000)

    # Compute distances
    chamfer_output = chamfer_distance(original_points, output_points)
    hausdorff_output = hausdorff_distance(original_points, output_points)
    chamfer_baseline = chamfer_distance(original_points, baseline_points)
    hausdorff_baseline = hausdorff_distance(original_points, baseline_points)

    # Voxelize for MSE
    voxel_size = 0.05  # Adjust as needed
    original_voxel = original_mesh.voxelized(voxel_size).matrix
    output_voxel = output_mesh.voxelized(voxel_size).matrix
    baseline_voxel = baseline_mesh.voxelized(voxel_size).matrix

    # Compute MSE
    mse_output = voxel_mse(original_voxel, output_voxel)
    mse_baseline = voxel_mse(original_voxel, baseline_voxel)

    print(f"Results for {model_name}:")
    print(f"Chamfer Distance (Output): {chamfer_output:.6f}")
    print(f"Hausdorff Distance (Output): {hausdorff_output:.6f}")
    print(f"Chamfer Distance (Baseline): {chamfer_baseline:.6f}")
    print(f"Hausdorff Distance (Baseline): {hausdorff_baseline:.6f}")
    print(f"MSE (Output): {mse_output:.6f}")
    print(f"MSE (Baseline): {mse_baseline:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Compare models for all or a custom output model.")
    parser.add_argument("--custom_output_path", type=str, help="Path to the custom output model (optional).")
    parser.add_argument("--custom_original_path", type=str, help="Path to the custom original model (optional).")
    parser.add_argument("--custom_baseline_path", type=str, help="Path to the custom baseline model (optional).")
    args = parser.parse_args()

    if args.custom_output_path:
        # Handle custom model
        if not (args.custom_original_path and args.custom_baseline_path):
            print("Error: You must provide paths for the original and baseline models with a custom output.")
            return

        compare_models(
            "custom_model",
            args.custom_output_path,
            args.custom_original_path,
            args.custom_baseline_path
        )
    else:
        # Iterate through the default configuration
        for model in default_config:
            if os.path.exists(model["output_path"]) and os.path.exists(model["original_path"]) and os.path.exists(model["baseline_path"]):
                compare_models(model["name"], model["output_path"], model["original_path"], model["baseline_path"])
            else:
                print(f"Skipping {model['name']}: Missing one or more files.")

if __name__ == "__main__":
    main()

"""
Script: model_comparison.py
Description: This script evaluates the quality of reconstructed 3D models by comparing them against original models and baseline models. It computes Chamfer Distance, Hausdorff Distance, and Mean Squared Error (MSE) metrics and provides the option to handle custom models.

Sources:
- **SciPy `cKDTree`**: Used for nearest-neighbor queries to compute Chamfer Distance and Hausdorff Distance.
  - Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
- **Trimesh**: Used for loading, sampling, and voxelizing 3D models.
  - URL: https://github.com/mikedh/trimesh
- **NumPy**: Used for numerical operations, including voxel alignment and logical operations.
  - Documentation: https://numpy.org/doc/

Contributors:
- Joshua Fantillo, John Ordoyo, Silas Myrrh: Developed and refined the script for automated comparison of multiple 3D models, integrating advanced voxel-based MSE computation and command-line interface for customization.

AI Assistance:
- OpenAI's ChatGPT contributed to the script's modularization, metric integration, and validation logic.

Usage Instructions:

To run the script for model comparisons, use the following command:

python model_comparison.py --custom_output_path path/to/reconstructed.obj \
                           --custom_original_path path/to/original.obj \
                           --custom_baseline_path path/to/baseline.obj

Citations for Libraries:

1. SciPy: For efficient computation of point-based distances using cKDTree.
   URL: https://www.scipy.org/

2. Trimesh: For handling 3D geometry operations such as loading, sampling, and voxelization.
   URL: https://github.com/mikedh/trimesh

3. NumPy: For performing numerical computations, including voxel alignment and logical operations.
   URL: https://numpy.org/
"""
