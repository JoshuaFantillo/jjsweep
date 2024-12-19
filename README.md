# 3D Model Processing and Comparison Pipeline

This repository contains scripts for processing, comparing, and visualizing 3D models using SphereNet and related methods. The pipeline supports both predefined models and custom models.

---

## **Setup**

### **Install Required Libraries**
Ensure the required libraries are listed in the `requirements.txt` file. Install them using:

```bash
pip install -r requirements.txt
```

### **Folder Structure**
Ensure your project directory has the following structure:

$ .
$ ├── data/                 # Directory containing original models  
$ │   ├── cat/  
$ │   │   └── cat_model.obj  
$ │   ├── dog/  
$ │   │   └── dog.obj  
$ │   └── ...  
$ ├── output/               # Directory for generated output  
$ ├── scripts/              # Directory containing the core scripts  
$ │   ├── mesh-collider.py  
$ │   └── compare.py  
$ ├── requirements.txt      # File for installing dependencies  
$ ├── run-all.py            # Script to process predefined/custom models  
$ ├── baseline-all.py       # Script to run the SphereNet baseline  
$ ├── compare-all.py        # Script to compare models  
$ └── show-all.py           # Script to visualize models  

---

## **How to Use**

### **Run Predefined Models**
To process the predefined models in the `data/` directory, run:
```bash
python run-all.py
```
### **Run Custom Models**
To process a custom model, specify the model path:
```bash
python run-all.py --new_model_path path/to/your_model.obj
```
### **Run SphereNet Baseline**
To process the baseline SphereNet implementation, run:
```bash
python baseline-all.py
```
### **Compare Models**
To compare predefined models, run:
```bash
python compare-all.py
```
To compare a custom model, specify the paths to the custom output, original, and baseline models:
```bash
python compare-all.py --custom_output_path path/to/output.obj --custom_original_path path/to/original.obj --custom_baseline_path path/to/baseline.obj
```
### **Visualize Models**
To visualize predefined models, run:
```bash
python show-all.py
```
To visualize a custom model, specify the paths:
```bash
python show-all.py --custom_output_path path/to/output.obj --custom_original_path path/to/original.obj --custom_baseline_path path/to/baseline.obj
```
---

## **Features**

- **Processing**: Voxelizes and processes 3D models into simplified representations using SphereNet.  
- **Comparison**: Computes Chamfer Distance, Hausdorff Distance, and MSE for model comparison.  
- **Visualization**: Displays 3D models for easy inspection.

---

## **Contributors**

- **Joshua Fantillo**  (josfantillo@gmail.com)
- **John Ordoyo**  (gtom_1984@hotmail.ca)
- **Silas Myrrh**  (silas.myrrh@gmail.com)

This implementation incorporates concepts and code adapted from CMPT464/764 assignments (Fall 2024) at Simon Fraser University, focusing on primitive shape abstraction and neural network-based spherical fitting techniques.
