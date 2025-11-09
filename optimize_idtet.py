# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
import os
from examples.util import *
import examples.render as render
import examples.loss as loss
import imageio
from basic_marching_tet.grid import VoxelTetGrid
from basic_marching_tet.mt import mt, mt_surface
from implicit_dtet.basic import delaunay
from basic_marching_tet.energy import compute_tet_mean_ratio, compute_min_dihedral_angle
from implicit_dtet.exp_1.common import MAX_TET_MEAN_RATIO
from implicit_dtet.iso.mt import VoxelVertexBandDownsampler
from implicit_dtet.basic import TetMesh
from matplotlib import pyplot as plt
from kaolin.ops.conversions import marching_tetrahedra

import sys
sys.path.append('..')
import tqdm

###############################################################################
# Functions adapted from https://github.com/NVlabs/nvdiffrec
###############################################################################

def lr_schedule(iter):
    return max(0.0, 10**(-(iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

def compute_tet_volume(verts: torch.Tensor, tets: torch.Tensor) -> torch.Tensor:
    # [M,4,3]
    P = verts[tets]

    # --- Volume (batched) ---
    # 6*V = det([p1-p0, p2-p0, p3-p0])
    p0, p1, p2, p3 = P[:,0], P[:,1], P[:,2], P[:,3]
    Mmat = torch.stack([p1 - p0, p2 - p0, p3 - p0], dim=-2)   # [M,3,3]
    det = torch.linalg.det(Mmat)                              # [M]
    vol = det / 6.0
    vol = vol.abs()                                    # [M]
    return vol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='idtet optimization')
    parser.add_argument('-o', '--out_dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)    
    
    parser.add_argument('-i0', '--iter_stage_0', type=int, default=1000)
    parser.add_argument('-i1', '--iter_stage_1', type=int, default=4000)
    parser.add_argument('-b', '--batch', type=int, default=8)
    parser.add_argument('-r', '--train_res', nargs=2, type=int, default=[2048, 2048])
    parser.add_argument('-lr0', '--learning_rate_stage_0', type=float, default=1e-2)
    parser.add_argument('-lr1', '--learning_rate_stage_1', type=float, default=1e-3)
    parser.add_argument('-minlr', '--min_learning_rate', type=float, default=1e-4)
    parser.add_argument('--voxel_grid_res', type=int, default=64)
    
    parser.add_argument('--sdf_loss', type=bool, default=True)
    parser.add_argument('--develop_reg', type=bool, default=False)
    parser.add_argument('--sdf_regularizer', type=float, default=0.2)
    parser.add_argument('--quality_regularizer', type=float, default=1e-3)
    parser.add_argument('--volume_regularizer', type=float, default=1e-1)
    
    parser.add_argument('-dr', '--display_res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-si', '--save_interval', type=int, default=20)
    FLAGS = parser.parse_args()
    device = 'cuda'
    
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    glctx = dr.RasterizeCudaContext()
    
    # Load GT mesh
    gt_mesh = load_mesh(FLAGS.ref_mesh, device)
    gt_mesh.auto_normals() # compute face normals for visualization
    
    # ==============================================================================================
    #  Create and initialize iDTet
    # ==============================================================================================
    # fc = FlexiCubes(device)
    domain_min = (-0.5, -0.5, -0.5)
    domain_max = (0.5, 0.5, 0.5)
    MT_VOXEL_GRID_RES = FLAGS.voxel_grid_res + 1        # +1 to get same number of vertices as FlexiCubes
    tet_grid = VoxelTetGrid(device=device)
    tet_grid.init(domain_min=domain_min, domain_max=domain_max, num_grid=MT_VOXEL_GRID_RES)
    verts = tet_grid.verts
    verts *= 2 # scale up the grid so that it's larger than the target object
    
    torch.manual_seed(0)
    sdf = torch.rand_like(verts[:,0]) - 0.1 # randomly init SDF
    sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
    # set per-cube learnable weights to zeros
    deform = torch.nn.Parameter(torch.zeros_like(verts), requires_grad=True)
    
    def train(stage, learning_rate, num_iter):
        out_dir = os.path.join(FLAGS.out_dir, f'stage_{stage}')
        os.makedirs(out_dir, exist_ok=True)
        
        # ==============================================================================================
        #  Setup optimizer
        # ==============================================================================================
        if stage == 0:
            # Preliminary stage: only optimize SDF
            optimizer = torch.optim.Adam([sdf], lr=learning_rate)
        else:
            # Full optimization: optimize SDF and vertex deformation
            optimizer = torch.optim.Adam([sdf, deform], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=FLAGS.min_learning_rate)
    
        if stage == 0:
            # Preliminary stage: only optimize SDF
            tets = delaunay(verts)
            
        losses = []
        r_losses = []
        q_losses = []
        
        # ==============================================================================================
        #  Train loop
        # ============================================================================================== 
        bar = tqdm.tqdm(range(num_iter))        
        for it in bar:
            optimizer.zero_grad()
            # sample random camera poses
            mv, mvp = render.get_random_camera_batch(FLAGS.batch, iter_res=FLAGS.train_res, device=device, use_kaolin=False)
            # render gt mesh
            target = render.render_mesh_paper(gt_mesh, mv, mvp, FLAGS.train_res, ["mask", "depth", "normal"])
            # extract and render idtet mesh
            grid_verts = verts
            if stage == 1:
                # deform and update tets
                grid_verts = verts + deform
                tets = delaunay(grid_verts)
            
            # if stage == 0:
            #     mt_verts, mt_faces = mt_surface(grid_verts, tets, sdf, iso=0.0)
            # else:
            mt_verts, mt_tets, mt_faces = mt(grid_verts, tets, sdf, iso=0.0)
            idtet_mesh = Mesh(mt_verts, mt_faces)
            idtet_mesh.auto_normals() # compute face normals for visualization
            buffers = render.render_mesh_paper(idtet_mesh, mv, mvp, FLAGS.train_res, ["mask", "depth", "normal"])
            
            # evaluate reconstruction loss
            mask_loss = (buffers['mask'] - target['mask']).abs().mean()
            depth_loss = (((((buffers['depth'] - (target['depth']))* target['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10
            normal_loss = ((buffers['normal'] - target['normal'])* target['mask']).abs().mean()
            recon_loss = mask_loss + depth_loss + normal_loss
            
            # evaluate quality loss
            if stage == 1:
                tets_mean_ratio = torch.clamp(compute_tet_mean_ratio(mt_verts, mt_tets), min=0.0, max=MAX_TET_MEAN_RATIO)
                quality_loss = ((MAX_TET_MEAN_RATIO - tets_mean_ratio) / MAX_TET_MEAN_RATIO).mean() * FLAGS.quality_regularizer
            else:
                quality_loss = torch.tensor(0.0, device=device)
                
            # evaluate volume loss: to remove floaters inside the shape
            tets_volume = compute_tet_volume(mt_verts, mt_tets)
            volume_loss = (1-tets_volume.sum()) * FLAGS.volume_regularizer
            
            total_loss = recon_loss + quality_loss + volume_loss
            
            # if FLAGS.sdf_loss: # optionally add SDF loss to eliminate internal structures
            #     with torch.no_grad():
            #         pts = sample_random_points(1000, gt_mesh)
            #         gt_sdf = compute_sdf(pts, gt_mesh.vertices, gt_mesh.faces)
            #     pred_sdf = compute_sdf(pts, idtet_mesh.vertices, idtet_mesh.faces)
            #     total_loss += torch.nn.functional.mse_loss(pred_sdf, gt_sdf) * 2e3
            
            losses.append(total_loss.item())
            r_losses.append(recon_loss.item())
            q_losses.append(quality_loss.item())
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()        
            bar.set_description(f"S{stage} | L: {total_loss.item():.4g} | RL: {(mask_loss + depth_loss).item():.4g} | QL: {quality_loss.item():.4g} | VL: {volume_loss.item():.4g} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if (it % FLAGS.save_interval == 0 or it == (num_iter-1)): # save normal image for visualization
                with torch.no_grad():
                    
                    grid_verts = verts 
                    if stage == 1:
                        grid_verts = verts + deform
                        tets = delaunay(grid_verts)
                    
                    # if stage == 0:
                    #     mt_verts, mt_faces = mt_surface(grid_verts, tets, sdf, iso=0.0)
                    # else:
                    mt_verts, mt_tets, mt_faces = mt(grid_verts, tets, sdf, iso=0.0)
                    
                    idtet_mesh = Mesh(mt_verts, mt_faces)
                    
                    idtet_mesh.auto_normals() # compute face normals for visualization
                    mv, mvp = render.get_rotate_camera(it//FLAGS.save_interval, iter_res=FLAGS.display_res, device=device,use_kaolin=False)
                    val_buffers = render.render_mesh_paper(idtet_mesh, mv.unsqueeze(0), mvp.unsqueeze(0), FLAGS.display_res, return_types=["normal"], white_bg=True)
                    val_image = ((val_buffers["normal"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
                    
                    gt_buffers = render.render_mesh_paper(gt_mesh, mv.unsqueeze(0), mvp.unsqueeze(0), FLAGS.display_res, return_types=["normal"], white_bg=True)
                    gt_image = ((gt_buffers["normal"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
                    imageio.imwrite(os.path.join(out_dir, '{:04d}.png'.format(it)), np.concatenate([val_image, gt_image], 1))
                    print(f"Optimization Step [{it}/{num_iter}], Loss: {total_loss.item():.4f}")
                    
                    # Compute tet dihedral angles
                    if stage == 1:
                        min_dihedral_angles, _ = compute_min_dihedral_angle(mt_verts, mt_tets, degrees=True)
                        ratio_0 = (min_dihedral_angles < 10.0).float().sum() / len(min_dihedral_angles)
                        plt.figure()
                        plt.hist(min_dihedral_angles.detach().cpu().numpy().flatten(), bins=30)
                        plt.xlabel('Minimum Dihedral Angle (degrees)')
                        plt.ylabel('Frequency')
                        plt.title(f'Histogram of Minimum Dihedral Angles (<10° ratio: {ratio_0:.4f})')
                        plt.savefig(os.path.join(out_dir, f'min_dihedral_angles.png'))
                        plt.close()
                
                    # ==============================================================================================
                    #  Save ouput
                    # ==============================================================================================     
                    mesh_np = trimesh.Trimesh(vertices = mt_verts.detach().cpu().numpy(), faces=mt_faces.detach().cpu().numpy(), process=False)
                    mesh_np.export(os.path.join(out_dir, 'output_mesh.obj'))
                    
                    # plot loss curve
                    plt.figure()
                    plt.plot(np.arange(len(losses)), losses, label='total loss')
                    plt.yscale('log')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.savefig(os.path.join(out_dir, 'total_loss_curve.png'))
                    plt.close()
                    
                    plt.figure()
                    plt.plot(np.arange(len(r_losses)), r_losses, label='reconstruction loss')
                    plt.yscale('log')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.savefig(os.path.join(out_dir, 'recon_loss_curve.png'))
                    plt.close()
                    
                    plt.figure()
                    plt.plot(np.arange(len(q_losses)), q_losses, label=f'quality loss ({q_losses[-1]:.4g})')
                    plt.yscale('log')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.savefig(os.path.join(out_dir, 'quality_loss_curve.png'))
                    plt.close()
                    
                    # save tetmesh
                    tetmesh = TetMesh(mt_verts, mt_tets)
                    tetmesh.export(out_dir)
    
    train(0, FLAGS.learning_rate_stage_0, FLAGS.iter_stage_0)
    
    # Downsample verts
    with torch.no_grad():
        downsampler = VoxelVertexBandDownsampler(tet_grid, iso=0.0)
        verts, mask = downsampler.forward(sdf)
        verts, sdf, deform = verts[mask], sdf[mask], deform[mask]
    verts = verts.clone().detach().requires_grad_(True)
    sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
    deform = torch.nn.Parameter(deform.clone().detach(), requires_grad=True)
    train(1, FLAGS.learning_rate_stage_1, FLAGS.iter_stage_1)