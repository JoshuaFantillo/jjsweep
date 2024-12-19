import trimesh
import argparse

# Default configuration for models
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

def visualize_individual_model(model_name, mesh_path, label):
    print(f"Displaying {label} model for {model_name}. Close the window to continue.")
    mesh = trimesh.load(mesh_path)
    mesh.show()

def visualize_models_sequentially(model_name, original_path, output_path, baseline_path):
    print(f"Visualizing models for {model_name}...")

    visualize_individual_model(model_name, original_path, "Original")

    visualize_individual_model(model_name, output_path, "Reconstructed")

    visualize_individual_model(model_name, baseline_path, "Baseline")

def main():
    parser = argparse.ArgumentParser(description="Visualize models for all or a custom model.")
    parser.add_argument("--custom_output_path", type=str, help="Path to the custom output model (optional).")
    parser.add_argument("--custom_original_path", type=str, help="Path to the custom original model (optional).")
    parser.add_argument("--custom_baseline_path", type=str, help="Path to the custom baseline model (optional).")
    args = parser.parse_args()

    if args.custom_output_path:
        if not (args.custom_original_path and args.custom_baseline_path):
            print("Error: You must provide paths for the original and baseline models with a custom output.")
            return

        visualize_models_sequentially(
            "custom_model",
            args.custom_original_path,
            args.custom_output_path,
            args.custom_baseline_path,
        )
    else:
        print("Visualizing all models from the default configuration...")
        for config in default_config:
            try:
                visualize_models_sequentially(
                    config["name"],
                    config["original_path"],
                    config["output_path"],
                    config["baseline_path"],
                )
            except Exception as e:
                print(f"Failed to visualize {config['name']}: {e}")

if __name__ == "__main__":
    main()

"""
Script: show-all.py
Description: This script visualizes 3D models sequentially, including the original, reconstructed, and baseline versions. It supports both predefined model configurations and custom models provided via command-line arguments.

Sources:
- Python `trimesh` library: Used for loading and displaying 3D models.
- Python `argparse` module: Used for command-line argument parsing.

Contributors:
- Joshua Fantillo, John Ordoyo, Silas Myrrh: Customized, debugged, and optimized the script for sequential visualization of models.

AI Assistance:
- Portions of this code were generated and refined using OpenAI's ChatGPT, specifically for implementing sequential visualization logic and error handling.

Usage:
- To visualize all predefined models, run the script without arguments.
- To visualize a custom model, provide `--custom_output_path`, `--custom_original_path`, and `--custom_baseline_path` as arguments.

Citations for Libraries:
- **Trimesh**: 
  - Trimesh: A Python library for loading and manipulating triangular meshes.
  - URL: https://github.com/mikedh/trimesh
"""
