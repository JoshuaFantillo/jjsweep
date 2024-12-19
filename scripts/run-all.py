import os
import subprocess
import argparse
import time

# Define the predefined models and their parameters
model_configs = [
    #{"name": "cat", "path": "data/cat/cat_model.obj", "k": 15, "grid_size": 0.04},
    #{"name": "dog", "path": "data/dog/dog.obj", "k": 20, "grid_size": 0.03},
    #{"name": "hand", "path": "data/hand/hand.obj", "k": 30, "grid_size": 0.01},
    #{"name": "pot", "path": "data/pot/pot.obj", "k": 45, "grid_size": 0.01},
    #{"name": "rod", "path": "data/rod/rod.obj", "k": 1, "grid_size": 0.01},
    #{"name": "shiba", "path": "data/shiba/shiba_model.obj", "k": 25, "grid_size": 0.03},
    #{"name": "sofa", "path": "data/sofa/sofa.obj", "k": 25, "grid_size": 0.02},
    #{"name": "starfish", "path": "data/starfish/starfish_model.obj", "k": 10, "grid_size": 0.03},
    {"name": "fertility", "path": "data/fertility/fertility.obj", "k": 25, "grid_size": 0.025},
    {"name": "suzanne", "path": "data/suzanne/suzanne.obj", "k": 25, "grid_size": 0.025},
    {"name": "torus_tri", "path": "data/torus_tri/torus_tri.obj", "k": 25, "grid_size": 0.025},
]

# Output folder for processed models
output_folder = "output"

def process_model(model_name, model_path, k, grid_size):
    print(f"Processing {model_name} with k={k} and grid_size={grid_size}...")

    # Call mesh-collider.py with subprocess and pass arguments
    result = subprocess.run(
        [
            "python", "scripts/mesh-collider.py",
            "--model_path", model_path,
            "--k", str(k),
            "--grid_size", str(grid_size),
            "--output_folder", os.path.join(output_folder, model_name)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"{model_name} processed successfully.")
    else:
        print(f"Error processing {model_name}:\n{result.stderr}")

# Used to run either pre-defined or custom model
def run_mesh_collider(new_model_path=None):
    if new_model_path:
        process_model("custom_model", new_model_path, k=20, grid_size=0.05)
    else:
        for config in model_configs:
            start_time = time.time()
            process_model(config["name"], config["path"], config["k"], config["grid_size"])
            elapsed_time = time.time() - start_time
            print(f"Total runtime " + config["name"] + ": " + str(elapsed_time) +" in seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mesh collider on predefined or custom models.")
    parser.add_argument("--new_model_path", type=str, help="Path to a new custom model to process.")
    args = parser.parse_args()

    run_mesh_collider(new_model_path=args.new_model_path)

"""
Script: run-all.py
Description: This script processes 3D models using the mesh-collider pipeline. It supports predefined model configurations or a custom model provided via command line arguments. Outputs are stored in the specified folder.

Sources:
- Python `subprocess` module: Used for executing external scripts.
- Trimesh library: https://github.com/mikedh/trimesh
- NumPy library: https://numpy.org/
- scikit-learn (KMeans clustering): https://scikit-learn.org/

Contributors:
- Joshua Fantillo, John Ordoyo, Silas Myrrh: Customized, debugged, and optimized the script for specific use cases.

Usage:
- To process predefined models, run the script without arguments.
- To process a custom model, provide the `--new_model_path` argument along with the path to the model file.

AI Assistance:
- Portions of this code were generated and refined using OpenAI's ChatGPT, specifically for implementing subprocess handling, runtime tracking, and debugging.

Citations for Libraries:
- **Trimesh**: 
  - Trimesh: A Python library for loading and manipulating triangular meshes.
  - URL: https://github.com/mikedh/trimesh
- **NumPy**: 
  - Harris et al., "Array programming with NumPy", *Nature*, 2020.
- **scikit-learn**:
  - Pedregosa et al., "Scikit-learn: Machine Learning in Python", *Journal of Machine Learning Research*, 2011.
"""
