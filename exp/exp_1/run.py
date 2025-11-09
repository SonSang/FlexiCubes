import os
import sys
import numpy as np

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

DATA_DIR = "/home/sanghyun/Documents/research/FlexiCubes/examples/data/inputmodels"
MESH_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.obj') and f in mesh_list]
MESH_FILES = sorted(MESH_FILES)
print("Found %d mesh files." % len(MESH_FILES))

for mesh_file in MESH_FILES:
    output_dir = f"output/exp_1/ours/{os.path.basename(mesh_file).split('.')[0]}"
    if os.path.exists(output_dir):
        print("Skipping %s" % mesh_file)
        continue
    print("Processing %s" % mesh_file)
    os.system(f"python optimize_idtet.py --out_dir output/exp_1/ours/{os.path.basename(mesh_file).split('.')[0]} --ref_mesh {mesh_file} --voxel_grid_res 96 -si 100 -i1 9000")