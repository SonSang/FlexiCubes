# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
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
import time
from PIL import Image

import sys
sys.path.append('..')
from flexicubes import FlexiCubes

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from test_renderer import AlphaRenderer, make_star_cameras, GTInitializer, calc_vertex_normals

###############################################################################
# Functions adapted from https://github.com/NVlabs/nvdiffrec
###############################################################################

def lr_schedule(iter):
    return max(0.0, 10**(-(iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='flexicubes optimization')
    parser.add_argument('-o', '--out_dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)    
    
    parser.add_argument('-i', '--iter', type=int, default=1000)
    parser.add_argument('-b', '--batch', type=int, default=8)
    parser.add_argument('-r', '--train_res', nargs=2, type=int, default=[256, 256])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--voxel_grid_res', type=int, default=64)
    
    parser.add_argument('--sdf_loss', type=bool, default=False)
    parser.add_argument('--develop_reg', type=bool, default=False)
    parser.add_argument('--sdf_regularizer', type=float, default=0.2)
    
    parser.add_argument('-dr', '--display_res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-si', '--save_interval', type=int, default=20)
    FLAGS = parser.parse_args()
    device = 'cuda:0'

    logdir = FLAGS.out_dir
    logdir = logdir + time.strftime("-%Y%m%d-%H%M%S")
    
    os.makedirs(logdir, exist_ok=True)
    glctx = dr.RasterizeGLContext()
    
    # Load GT mesh
    gt_mesh = load_mesh(FLAGS.ref_mesh, device)
    
    target_vertices = gt_mesh.vertices - gt_mesh.vertices.mean(dim=0, keepdim=True)
    max_norm = torch.max(torch.norm(target_vertices, dim=-1)) + 1e-6
    target_vertices = (target_vertices / max_norm) * 1.0
    gt_mesh.vertices = target_vertices

    gt_mesh.auto_normals() # compute face normals for visualization
    
    # ==============================================================================================
    #  Create and initialize FlexiCubes
    # ==============================================================================================
    fc = FlexiCubes(device)
    x_nx3, cube_fx8 = fc.construct_voxel_grid(FLAGS.voxel_grid_res)
    x_nx3 *= 2 # scale up the grid so that it's larger than the target object
    
    sdf = torch.rand_like(x_nx3[:,0]) - 0.1 # randomly init SDF
    sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
    # set per-cube learnable weights to zeros
    weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device='cuda') 
    weight    = torch.nn.Parameter(weight.clone().detach(), requires_grad=True)
    deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)
    
    #  Retrieve all the edges of the voxel grid; these edges will be utilized to 
    #  compute the regularization loss in subsequent steps of the process.    
    all_edges = cube_fx8[:, fc.cube_edges].reshape(-1, 2)
    grid_edges = torch.unique(all_edges, dim=0)
    
    # ==============================================================================================
    #  Setup optimizer
    # ==============================================================================================
    optimizer = torch.optim.Adam([sdf, weight,deform], lr=FLAGS.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x)) 
    
    # cameras;
    num_viewpoints = 8
    image_size = 256
    mv, proj = make_star_cameras(num_viewpoints, num_viewpoints, distance=2.0, r=0.6, n=1.0, f=3.0)
    renderer = AlphaRenderer(mv, proj, [image_size, image_size])    

    gt_manager = GTInitializer(gt_mesh.vertices, gt_mesh.faces, device)
    gt_manager.render(renderer)

    gt_diffuse_map = gt_manager.diffuse_images()
    gt_depth_map = gt_manager.depth_images()
    gt_shil_map = gt_manager.shillouette_images()

    # save gt images;
    image_save_path = os.path.join(logdir, "gt_images")
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
        
    def save_image(img, path):
        img = img.cpu().numpy()
        img = img * 255.0
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img.save(path)

    for i in range(len(gt_diffuse_map)):
        save_image(gt_diffuse_map[i], os.path.join(image_save_path, "diffuse_{}.png".format(i)))
        save_image(gt_depth_map[i], os.path.join(image_save_path, "depth_{}.png".format(i)))
        save_image(gt_shil_map[i], os.path.join(image_save_path, "shil_{}.png".format(i)))

    writer = SummaryWriter(logdir)

    # ==============================================================================================
    #  Train loop
    # ==============================================================================================   
    bar = tqdm(range(FLAGS.iter))
    for it in bar: 
        optimizer.zero_grad()
        # sample random camera poses
        # mv, mvp = render.get_random_camera_batch(FLAGS.batch, iter_res=FLAGS.train_res, device=device)
        # render gt mesh
        # target = render.render_mesh(gt_mesh, mv, mvp, FLAGS.train_res)
        # extract and render FlexiCubes mesh
        grid_verts = x_nx3 + (2-1e-8) / (FLAGS.voxel_grid_res * 2) * torch.tanh(deform)
        vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, FLAGS.voxel_grid_res, beta_fx12=weight[:,:12], alpha_fx8=weight[:,12:20],
            gamma_f=weight[:,20], training=True)
        flexicubes_mesh = Mesh(vertices, faces)
        # buffers = render.render_mesh(flexicubes_mesh, mv, mvp, FLAGS.train_res)

        # render our mesh;
        fvertices = flexicubes_mesh.vertices
        ffaces = flexicubes_mesh.faces
        fvertex_normals = calc_vertex_normals(fvertices, ffaces)

        col, _ = renderer.forward(
            fvertices, fvertex_normals, ffaces)
        diffuse = col[...,:-2]
        depth = col[..., [-2, -2, -2]]
        
        # evaluate reconstruction loss
        diffuse_loss = torch.abs(diffuse - gt_diffuse_map).mean()
        depth_loss = torch.abs(depth - gt_depth_map).mean()
        # mask_loss = (buffers['mask'] - target['mask']).abs().mean()
        # depth_loss = (((((buffers['depth'] - (target['depth']))* target['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10
    
        t_iter = it / FLAGS.iter
        sdf_weight = FLAGS.sdf_regularizer - (FLAGS.sdf_regularizer - FLAGS.sdf_regularizer/20)*min(1.0, 4.0 * t_iter)
        reg_loss = loss.sdf_reg_loss(sdf, grid_edges).mean() * sdf_weight # Loss to eliminate internal floaters that are not visible
        reg_loss += L_dev.mean() * 0.5
        reg_loss += (weight[:,:20]).abs().mean() * 0.1
        total_loss = diffuse_loss + depth_loss + reg_loss
        
        if False: #FLAGS.sdf_loss: # optionally add SDF loss to eliminate internal structures
            with torch.no_grad():
                pts = sample_random_points(1000, gt_mesh)
                gt_sdf = compute_sdf(pts, gt_mesh.vertices, gt_mesh.faces)
            pred_sdf = compute_sdf(pts, flexicubes_mesh.vertices, flexicubes_mesh.faces)
            total_loss += torch.nn.functional.mse_loss(pred_sdf, gt_sdf) * 2e3
        
        # optionally add developability regularizer, as described in paper section 5.2
        if FLAGS.develop_reg:
            reg_weight = max(0, t_iter - 0.8) * 5
            if reg_weight > 0: # only applied after shape converges
                reg_loss = loss.mesh_developable_reg(flexicubes_mesh).mean() * 10
                reg_loss += (deform).abs().mean()
                reg_loss += (weight[:,:20]).abs().mean()
                total_loss = diffuse_loss + depth_loss + reg_loss 
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()        

        writer.add_scalar("loss/diffuse_loss", diffuse_loss, it)
        writer.add_scalar("loss/depth_loss", depth_loss, it)
        writer.add_scalar("loss/reg_loss", reg_loss, it)
        writer.add_scalar("loss/total_loss", total_loss, it)

        if (it % FLAGS.save_interval == 0 or it == (FLAGS.iter-1)): # save normal image for visualization
            writer.add_image("diffuse", diffuse[0], it, dataformats="HWC")
            writer.add_image("depth", depth[0], it, dataformats="HWC")

        bar.set_description("loss: {}".format(total_loss.item()))
        
        if (it % FLAGS.save_interval == 0 or it == (FLAGS.iter-1)): # save normal image for visualization
            with torch.no_grad():
                # extract mesh with training=False
                vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, FLAGS.voxel_grid_res, beta_fx12=weight[:,:12], alpha_fx8=weight[:,12:20],
                gamma_f=weight[:,20], training=False)
                flexicubes_mesh = Mesh(vertices, faces)

                save_mesh = trimesh.base.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
                save_mesh.export(os.path.join(logdir, '{:04d}.obj'.format(it)))
                
                # flexicubes_mesh.auto_normals() # compute face normals for visualization
                # mv, mvp = render.get_rotate_camera(it//FLAGS.save_interval, iter_res=FLAGS.display_res, device=device)
                # val_buffers = render.render_mesh(flexicubes_mesh, mv.unsqueeze(0), mvp.unsqueeze(0), FLAGS.display_res, return_types=["normal"], white_bg=True)
                # val_image = ((val_buffers["normal"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
                
        #         gt_buffers = render.render_mesh(gt_mesh, mv.unsqueeze(0), mvp.unsqueeze(0), FLAGS.display_res, return_types=["normal"], white_bg=True)
        #         gt_image = ((gt_buffers["normal"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
        #         imageio.imwrite(os.path.join(FLAGS.out_dir, '{:04d}.png'.format(it)), np.concatenate([val_image, gt_image], 1))
        #         print(f"Optimization Step [{it}/{FLAGS.iter}], Loss: {total_loss.item():.4f}")
            
    # ==============================================================================================
    #  Save ouput
    # ==============================================================================================     
    mesh_np = trimesh.Trimesh(vertices = vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
    mesh_np.export(os.path.join(logdir, 'output_mesh.obj'))