from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from pathlib import Path
import torch
import os

from nds.core import (
    Mesh, Renderer
)
from nds.losses import (
    mask_loss, normal_consistency_loss, laplacian_loss, shading_loss, depth_loss, normal_loss
)
from nds.modules import (
    SpaceNormalization, NeuralShader, ViewSampler
)
from nds.utils import (
    AABB, read_views, read_shaded_views, read_mesh, write_mesh, visualize_mesh_as_overlay, visualize_views, generate_mesh, mesh_generator_names
)
from nds.losses.shading import RandomSamples

from torchmetrics import PeakSignalNoiseRatio

def calculate_psnr_score(view, shaded_view, score_function=PeakSignalNoiseRatio(), shading_percentage=1):
    psnr = 0
    sample_fn = lambda x: x
    # Get valid area
    mask = ((view.mask > 0) & (shaded_view.mask > 0)).squeeze()

    # Sample only within valid area
    if shading_percentage != 1:
        sample_fn = RandomSamples(view.mask[mask].shape[0], 1, shading_percentage)
    target = sample_fn(view.color[mask])
    shaded = sample_fn(shaded_view.color[mask])

    psnr = score_function(shaded, target)
    return psnr 

if __name__ == "__main__":
    parser = ArgumentParser(description='Shading evaluation of the Neural Deffered Shading model with PSNR', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=Path, default="./data", help="Path to the input views")
    parser.add_argument('--input_sh_dir', type=Path, default="./data", help="Path to the shaded views")
    parser.add_argument('--input_bbox', type=Path, default=None, help="Path to the input bounding box. If None, it is computed from the input mesh")
    parser.add_argument('--output_dir', type=Path, default="./eval_results", help="Path to the output directory")
    parser.add_argument('--image_scale', type=int, default=1, help="Scale applied to the input images. The factor is 1/image_scale, so image_scale=2 halves the image size")
    parser.add_argument('--device', type=int, default=0, choices=([-1] + list(range(torch.cuda.device_count()))), help="GPU to use; -1 is CPU")
    parser.add_argument('--shading_percentage', type=float, default=0.75, help="Percentage of valid pixels considered in the shading loss (0-1)")
    parser.add_argument('--num_views', type=int, default=-1, help="Number of input views chosen at random from the input_dir")

    args = parser.parse_args()
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")
    views = read_views(args.input_dir, args.num_views, scale=args.image_scale, device=device)
    shaded_views = read_shaded_views(args.input_sh_dir, args.input_dir, args.num_views, scale=args.image_scale, device=device)
    # Load the bounding box or create it from the mesh vertices
    if args.input_bbox is not None:
        aabb = AABB.load(args.input_bbox)
    else:
        aabb = AABB(mesh.vertices.cpu().numpy())

    # Apply the normalizing affine transform, which maps the bounding box to 
    # a 2-cube centered at (0, 0, 0), to the views, the mesh, and the bounding box
    space_normalization = SpaceNormalization(aabb.corners)
    views = space_normalization.normalize_views(views)
    aabb = space_normalization.normalize_aabb(aabb)
    shaded_views = space_normalization.normalize_views(shaded_views)
    # Configure the renderer

    sh_loss = 0
    psnr = PeakSignalNoiseRatio().to(device)
    psnr_score = 0
    for view, shaded_view in zip(views, shaded_views):
        #sh_loss += shading_loss([view], [gbuffer], shader=shader, shading_percentage=args.shading_percentage).cpu().item()
        psnr_score += calculate_psnr_score(view, shaded_view, psnr, args.shading_percentage).cpu().item()
    #sh_loss = sh_loss / len(views)
    psnr_score = psnr_score / len(views)
    print("psnr score: %s" % psnr_score)
    instance_name = str(args.input_sh_dir).split('/')[-3]
    #path = '%s/shading_eval.txt' % args.output_dir
    path = '%s/%s_dtu_shading_eval.txt' % (args.output_dir, instance_name)
    isExist = os.path.exists(args.output_dir)
    if not isExist:
        os.makedirs(args.output_dir)
        print("The new directory is created!")

    with open(path, 'w') as f:
        f.write('shading loss: %s \npsnr score: %s' % (sh_loss, psnr_score))
