import numpy as np
from pathlib import Path
import trimesh
import random

from nds.core import Mesh, View

def read_mesh(path, device='cpu'):
    mesh_ = trimesh.load_mesh(str(path), process=False)

    vertices = np.array(mesh_.vertices, dtype=np.float32)
    indices = None
    if hasattr(mesh_, 'faces'):
        indices = np.array(mesh_.faces, dtype=np.int32)

    return Mesh(vertices, indices, device)

def write_mesh(path, mesh):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = mesh.vertices.numpy()
    indices = mesh.indices.numpy() if mesh.indices is not None else None
    mesh_ = trimesh.Trimesh(vertices=vertices, faces=indices, process=False)
    mesh_.export(path)

def read_views(directory, num_views, scale, device):
    directory = Path(directory)

    image_paths = sorted([path for path in directory.iterdir() if (path.is_file() and path.suffix == '.png')])
    
    views = []
    for image_path in image_paths:
        if 'normal' not in str(image_path):
            views.append(View.load(image_path, device))
    print("Found {:d} views".format(len(views)))

    if scale > 1:
        for view in views:
            view.scale(scale)
        print("Scaled views to 1/{:d}th size".format(scale))

    return views

def read_views(directory, num_views, scale, device):
    directory = Path(directory)

    image_paths = sorted([path for path in directory.iterdir() if (path.is_file() and path.suffix == '.png')])
    
    views = []
    for image_path in image_paths:
        if 'normal' not in str(image_path):
            views.append(View.load(image_path, device))
    
    # Ensure there are enough views to sample from
    if num_views != -1:
        if len(views) < num_views:
            raise ValueError(f"Error: Requested {num_views} views, but only found {len(views)}")
        views = random.sample(views, num_views)

    print("Found {:d} views".format(len(views)))

    if scale > 1:
        for view in views:
            view.scale(scale)
        print("Scaled views to 1/{:d}th size".format(scale))

    return views