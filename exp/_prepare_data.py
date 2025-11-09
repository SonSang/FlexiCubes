import os
import mesh2sdf
import trimesh
import numpy as np
import time
from tqdm import tqdm
from time import sleep

DATASET_DIR = "/home/sanghyun/Documents/research/FlexiCubes/examples/data/inputmodels"
filenames = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith('.obj')]
filenames = sorted(filenames)

OUTPUT_DIR = "/home/sanghyun/Documents/research/FlexiCubes/data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

mesh_scale = 1.0
size = 384
level = 2 / size

mesh_list = ["armadillo.obj", 
             "bimba100K.obj", 
             "botijo.obj", 
             "bunnyBotsch.obj", 
             "camel.obj", 
             "camille_hand100K.obj", 
             "cow2.obj", 
             "dancing_children100K", 
             "elephant.obj", 
             "fertility_tri.obj", 
             "gargoyle100K.obj", 
             "gearbox.obj", 
             "grayloc.obj"
            ]
bar = tqdm(filenames)
for filename in bar:    
    file_id = filename.split('/')[-1]
    if file_id not in mesh_list:
        continue
    output_name = os.path.join(OUTPUT_DIR, file_id)
    
    bar.set_description("Processing %s" % file_id)
    
    mesh = trimesh.load(filename)

    # normalize mesh
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * 0.95 / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    t0 = time.time()
    sdf, mesh = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
    t1 = time.time()
    
    mesh.vertices = mesh.vertices * mesh_scale
    
    mesh.export(output_name)