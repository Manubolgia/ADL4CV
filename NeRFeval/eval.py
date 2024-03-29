import argparse

import torch
import numpy as np

# pip install trimesh[all]
import trimesh
import os
from metrics import calculate_fscore
import open3d

# https://github.com/otaheri/chamfer_distance
from chamfer_distance import ChamferDistance

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values()))
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh

def sample_mesh(m, n):
    vpos, _ = trimesh.sample.sample_surface(m, n)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(vpos)
    return torch.tensor(vpos, dtype=torch.float32, device="cuda"), pcd

if __name__ == "__main__":
    chamfer_dist = ChamferDistance()

    parser = argparse.ArgumentParser(description='Chamfer loss')
    parser.add_argument('-n', type=int, default=2500000)
    parser.add_argument('--mesh', type=str)
    parser.add_argument('--ref', type=str)
    parser.add_argument('--out', type=str, default="eval_results/")
    parser.add_argument('--iter', type=int)
    FLAGS = parser.parse_args()

    mesh_path = FLAGS.mesh + "/mesh_%06d.obj" % FLAGS.iter
    mesh = as_mesh(trimesh.load(mesh_path))
    ref = as_mesh(trimesh.load(FLAGS.ref))

    print("Loaded meshes")

    # Make sure l=1.0 maps to 1/10th of the AABB. https://arxiv.org/pdf/1612.00603.pdf
    scale = 10.0 / np.amax(np.amax(ref.vertices, axis=0) - np.amin(ref.vertices, axis=0))
    ref.vertices = ref.vertices * scale
    mesh.vertices = mesh.vertices * scale

    # Sample mesh surfaces
    vpos_mesh, pcd_mesh = sample_mesh(mesh, FLAGS.n)
    vpos_ref, pcd_ref = sample_mesh(ref, FLAGS.n)

    print("Sampled meshes")

    dist1, dist2, idx1, idx2 = chamfer_dist(vpos_mesh[None, ...], vpos_ref[None, ...])
    loss = (torch.mean(dist1) + torch.mean(dist2)).item()
    fscore, precision, recall = calculate_fscore(pcd_ref, pcd_mesh)
    instance_name = FLAGS.mesh.split('/')[-2]

    path = '%s/%s_mesh_eval.txt' % (FLAGS.out, instance_name)
    isExist = os.path.exists(FLAGS.out)
    if not isExist:
        os.makedirs(FLAGS.out)
        print("The new directory is created!")

    with open(path, 'w') as f:
        f.write('dist1: %s \ndist2: %s \nidx1: %s \nidx2: %s \nloss: %f \ntris: %s \n\nfscore: %s \nprecision: %s \nrecall: %s' 
                % (torch.mean(dist1).item(), torch.mean(dist2).item(), idx1.float().mean().item(), idx2.float().mean().item(), loss, mesh.faces.shape[0], fscore, precision, recall))