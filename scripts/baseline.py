import torch
import numpy as np
import os
import trimesh
from trimesh import Scene, creation
import torch.nn as nn
import time



class DGCNNFeat(nn.Module):
    def __init__(self, k=20, emb_dims=256, dropout=0.5, global_feat=True):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout
        self.global_feat = global_feat
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=False)   
        x = self.conv1(x)                       
        x = self.conv2(x)                       
        x1 = x.max(dim=-1, keepdim=False)[0]   

        x = get_graph_feature(x1, k=self.k)    
        x = self.conv3(x)                       
        x = self.conv4(x)                      
        x2 = x.max(dim=-1, keepdim=False)[0]    

        x = get_graph_feature(x2, k=self.k)     
        x = self.conv5(x)                       
        x3 = x.max(dim=-1, keepdim=False)[0]   

        x = torch.cat((x1, x2, x3), dim=1)      

        x = self.conv6(x)                     
        if self.global_feat:
            x = x.max(dim=-1)[0]               
        return x                                

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        in_ch = 256
        out_ch = 1024
        feat_ch = 512

        self.net1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
            nn.ReLU(inplace=True),
        )

        self.net2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, out_ch),
        )
        num_params = sum(p.numel() for p in self.parameters())
        print("[num parameters: {}]".format(num_params))

    def forward(self, z):
        in1 = z
        out1 = self.net1(in1)
        in2 = torch.cat([out1, in1], dim=-1)
        out2 = self.net2(in2)
        return out2

class SphereNet(torch.nn.Module):
    def __init__(self, num_spheres=256):
        super(SphereNet, self).__init__()
        self.num_spheres = num_spheres
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder()

    def forward(self, surface_points, query_points):
        features = self.encoder(surface_points)
        sphere_params = self.decoder(features)
        sphere_params = torch.sigmoid(sphere_params.view(-1, 4))
        sphere_adder = torch.tensor([-0.5, -0.5, -0.5, 0.1]).to(sphere_params.device)
        sphere_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.4]).to(sphere_params.device)
        sphere_params = sphere_params * sphere_multiplier + sphere_adder
        sphere_sdf = determine_sphere_sdf(query_points, sphere_params)
        return sphere_sdf, sphere_params

def determine_sphere_sdf(query_points, sphere_params):
    sphere_centers = sphere_params[:, :3]
    sphere_radii = sphere_params[:, 3]
    diff = query_points[:, None, :] - sphere_centers[None, :, :]
    distances = torch.norm(diff, dim=2)
    sphere_sdf = distances - sphere_radii
    return sphere_sdf

def smoothness_loss(sphere_params):
    centers = sphere_params[:, :3]
    radii = sphere_params[:, 3]
    center_differences = torch.norm(centers[1:] - centers[:-1], dim=1)
    radius_differences = torch.abs(radii[1:] - radii[:-1])
    smooth_loss = torch.mean(center_differences + radius_differences)
    return smooth_loss

def l2_regularization(sphere_params):
    return torch.mean(sphere_params ** 2)

def visualise_spheres(sphere_params, save_path=None):
    sphere_params = sphere_params.cpu().detach().numpy()
    sphere_centers = sphere_params[..., :3]
    sphere_radii = np.abs(sphere_params[..., 3])
    scene = Scene()
    for center, radius in zip(sphere_centers, sphere_radii):
        sphere = creation.icosphere(radius=radius, subdivisions=2)
        sphere.apply_translation(center)
        scene.add_geometry(sphere)
    if save_path:
        scene.export(save_path)

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   
        else:
            idx = knn(x[:, 6:], k=k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature     

def voxelize_model(model_path, voxel_size=0.02):
    """Load and voxelize the 3D model."""
    print(f"Loading model from {model_path}...")
    mesh = trimesh.load(model_path)
    voxelized = mesh.voxelized(voxel_size)
    voxel_points = voxelized.points
    return voxel_points

def run_sphere_net_for_cat(model_path, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    voxel_points = voxelize_model(model_path)
    print(f"Voxelized model contains {len(voxel_points)} points.")

    model = SphereNet(num_spheres=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    mse_loss_fn = torch.nn.MSELoss()

    smooth_loss_weight = 0.1
    l2_loss_weight = 0.05
    num_epochs = 500

    surface_points = torch.from_numpy(voxel_points).float().to(device)
    surface_points = surface_points.unsqueeze(0).transpose(2, 1)

    query_points = torch.from_numpy(voxel_points).float().to(device)
    values = torch.zeros(query_points.shape[0]).to(device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        sphere_sdf, sphere_params = model(surface_points, query_points)
        sphere_sdf = torch.min(sphere_sdf, dim=-1).values
        mse_loss = mse_loss_fn(sphere_sdf, values)
        smooth_loss = smoothness_loss(sphere_params)
        l2_loss = l2_regularization(sphere_params)
        total_loss = mse_loss + smooth_loss_weight * smooth_loss + l2_loss_weight * l2_loss
        total_loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}")

    output_path = os.path.join(output_folder, "cat_sphere_params.npy")
    np.save(output_path, sphere_params.cpu().detach().numpy())
    output_model_path = os.path.join(output_folder, "cat_spheres.obj")
    visualise_spheres(sphere_params, save_path=output_model_path)

    print(f"Baseline processing complete. Results saved to {output_folder}")

import time

def main():
    model_path = "data/cat/cat_model.obj" 
    output_folder = "output"

    start_time = time.time()

    run_sphere_net_for_cat(model_path, output_folder)

    elapsed_time = time.time() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()

"""
Script: SphereNet Implementation for Shape Abstraction
Description: This script implements a neural network-based approach to abstract 3D models into spherical primitives. It is based on the SphereNet architecture, with an encoder-decoder framework, and includes additional loss functions for smoothness and regularization. Outputs include visualized sphere abstractions saved as 3D files.

Citation for Code Adaptation and Context:
The provided script adapts concepts and implementations from the CMPT464/764 Assignment 1 tasks, particularly focusing on Task 3 - Shape Abstraction with Neural Networks. The following components can be attributed to the assignment materials:
- **SphereNet Architecture**: The implementation of SphereNet, including its encoder (`DGCNNFeat`) and decoder, is inspired by Task 3 of the assignment.
- **Determine Sphere SDF Function**: Replicates the computation of signed distance fields for spheres, aligning with assignment requirements.
- **Visualization Techniques**: Aligns with instructions for visualizing spherical abstractions in Task 3.
- **Additional Neural Network Losses**: Incorporates smoothness and L2 regularization loss functions inspired by assignment bonus tasks.

Adapted Libraries and Techniques:
- **Trimesh**:
  - Purpose: Loading, voxelizing, and visualizing 3D models.
  - Reference: [Trimesh GitHub Repository](https://github.com/mikedh/trimesh)
- **PyTorch**:
  - Purpose: Neural network implementation and optimization.
  - Reference: [PyTorch Official Documentation](https://pytorch.org/)
- **NumPy**:
  - Purpose: Numerical computations and transformations for model inputs/outputs.
  - Reference: [NumPy Official Documentation](https://numpy.org/)
- **Matplotlib**:
  - Purpose: Auxiliary visualization functions for debugging and analysis.
  - Reference: [Matplotlib Official Website](https://matplotlib.org/)

Contributors:
- Joshua Fantillo, John Ordoyo, Silas Myrrh: Customized, debugged, and optimized the implementation for specific use cases, based on provided assignment instructions.
"""
