from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import meshzoo
import numpy as np
from pathlib import Path
from pyremesh import remesh_botsch
import torch
from tqdm import tqdm

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
    AABB, read_views, read_mesh, write_mesh, visualize_mesh_as_overlay, visualize_views, generate_mesh, mesh_generator_names
)

from nds.utils.debug import save_losses
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = ArgumentParser(description='Multi-View Mesh Reconstruction with Neural Deferred Shading', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=Path, default="./data", help="Path to the input data")
    parser.add_argument('--input_bbox', type=Path, default=None, help="Path to the input bounding box. If None, it is computed from the input mesh")
    parser.add_argument('--output_dir', type=Path, default="./out", help="Path to the output directory")
    parser.add_argument('--initial_mesh', type=str, default="vh32", help="Initial mesh, either a path or one of [vh16, vh32, vh64, sphere16]")
    parser.add_argument('--image_scale', type=int, default=1, help="Scale applied to the input images. The factor is 1/image_scale, so image_scale=2 halves the image size")
    parser.add_argument('--iterations', type=int, default=2000, help="Total number of iterations")
    parser.add_argument('--run_name', type=str, default=None, help="Name of this run")
    parser.add_argument('--lr_vertices', type=float, default=1e-3, help="Step size/learning rate for the vertex positions")
    parser.add_argument('--lr_shader', type=float, default=1e-3, help="Step size/learning rate for the shader parameters")
    parser.add_argument('--upsample_iterations', type=int, nargs='+', default=[500, 1000, 1500], help="Iterations at which to perform mesh upsampling")
    parser.add_argument('--save_frequency', type=int, default=100, help="Frequency of mesh and shader saving (in iterations)")
    parser.add_argument('--visualization_frequency', type=int, default=100, help="Frequency of shader visualization (in iterations)")
    parser.add_argument('--visualization_views', type=int, nargs='+', default=[], help="Views to use for visualization. By default, a random view is selected each time")
    parser.add_argument('--device', type=int, default=0, choices=([-1] + list(range(torch.cuda.device_count()))), help="GPU to use; -1 is CPU")
    parser.add_argument('--weight_mask', type=float, default=2.0, help="Weight of the mask term")
    parser.add_argument('--weight_normal_c', type=float, default=0.1, help="Weight of the normal consistency term")
    parser.add_argument('--weight_laplacian', type=float, default=40.0, help="Weight of the laplacian term")
    parser.add_argument('--weight_shading', type=float, default=1.0, help="Weight of the shading term")
    parser.add_argument('--weight_depth', type=float, default=0.0, help="Weight of the depth term")
    parser.add_argument('--weight_normal', type=float, default=0.0, help="Weight of the normal term")
    parser.add_argument('--shading_percentage', type=float, default=0.75, help="Percentage of valid pixels considered in the shading loss (0-1)")
    parser.add_argument('--hidden_features_layers', type=int, default=3, help="Number of hidden layers in the positional feature part of the neural shader")
    parser.add_argument('--hidden_features_size', type=int, default=256, help="Width of the hidden layers in the neural shader")
    parser.add_argument('--fourier_features', type=str, default='positional', choices=(['none', 'gfft', 'positional']), help="Input encoding used in the neural shader")
    parser.add_argument('--activation', type=str, default='relu', choices=(['relu', 'sine']), help="Activation function used in the neural shader")
    parser.add_argument('--fft_scale', type=int, default=4, help="Scale parameter of frequency-based input encodings in the neural shader")
    parser.add_argument('--num_views', type=int, default=-1, help="Number of input views chosen at random from the input_dir")
    parser.add_argument('--loss', type=str, default='L1', help="Choose between L1 and L2 loss, default is L1")
    parser.add_argument('--compare_size', type=int, default=200, help="Re-scaled size of normals and depth images used during loss calculation")
    # Add module arguments
    ViewSampler.add_arguments(parser)

    args = parser.parse_args()

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    # Create directories
    run_name = args.run_name if args.run_name is not None else args.input_dir.parent.name
    experiment_dir = args.output_dir / run_name

    images_save_path = experiment_dir / "images"
    meshes_save_path = experiment_dir / "meshes"
    shaders_save_path = experiment_dir / "shaders"
    plots_save_path = experiment_dir / "plots"
    images_save_path.mkdir(parents=True, exist_ok=True)
    meshes_save_path.mkdir(parents=True, exist_ok=True)
    shaders_save_path.mkdir(parents=True, exist_ok=True)
    plots_save_path.mkdir(parents=True, exist_ok=True)

    # Save args for this execution
    with open(experiment_dir / "args.txt", "w") as text_file:
        print(f"{args}", file=text_file)

    # Read the views
    views = read_views(args.input_dir, args.num_views, scale=args.image_scale, device=device)

    # Obtain the initial mesh and compute its connectivity
    mesh_initial: Mesh = None
    if args.initial_mesh in mesh_generator_names:
        # Use args.initial_mesh as mesh generator name
        if args.input_bbox is None:
            raise RuntimeError("Generated meshes require a bounding box.")
        mesh_initial = generate_mesh(args.initial_mesh, views, AABB.load(args.input_bbox), device=device)
    else:
        # Use args.initial_mesh as path to the mesh
        mesh_initial = read_mesh(args.initial_mesh, device=device)
    mesh_initial.compute_connectivity()

    # Load the bounding box or create it from the mesh vertices
    if args.input_bbox is not None:
        aabb = AABB.load(args.input_bbox)
    else:
        aabb = AABB(mesh_initial.vertices.cpu().numpy())
    aabb.save(experiment_dir / "bbox.txt")

    # Apply the normalizing affine transform, which maps the bounding box to 
    # a 2-cube centered at (0, 0, 0), to the views, the mesh, and the bounding box
    space_normalization = SpaceNormalization(aabb.corners)
    views = space_normalization.normalize_views(views)
    mesh_initial = space_normalization.normalize_mesh(mesh_initial)
    aabb = space_normalization.normalize_aabb(aabb)

    # Configure the renderer
    renderer = Renderer(device=device)
    renderer.set_near_far(views, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    # Visualize the inputs before optimization
    visualize_views(views, show=False, save_path=experiment_dir / "views.png")
    visualize_mesh_as_overlay(renderer, views, mesh_initial, show=False, save_path=experiment_dir / "views_overlay.png")

    # Configure the view sampler
    view_sampler = ViewSampler(views=views, **ViewSampler.get_parameters(args))
    # Create the optimizer for the vertex positions 
    # (we optimize offsets from the initial vertex position)
    lr_vertices = args.lr_vertices
    vertex_offsets = torch.zeros_like(mesh_initial.vertices)
    vertex_offsets.requires_grad = True
    optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)

    # Create the optimizer for the neural shader
    shader = NeuralShader(hidden_features_layers=args.hidden_features_layers,
                          hidden_features_size=args.hidden_features_size,
                          fourier_features=args.fourier_features,
                          activation=args.activation,
                          fft_scale=args.fft_scale,
                          last_activation=torch.nn.Sigmoid, 
                          device=device)
    optimizer_shader = torch.optim.Adam(shader.parameters(), lr=args.lr_shader)

    # Initialize the loss weights and losses
    loss_weights = {
        "mask": args.weight_mask,
        "normal": args.weight_normal,
        "depth": args.weight_depth,
        "normal_c":args.weight_normal_c,
        "laplacian": args.weight_laplacian,
        "shading": args.weight_shading
    }
    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}
    losses_record = {'mask': [], 'normal': [], 'depth': [], 'normal_c': [], 'laplacian': [], 'shading': []}

    progress_bar = tqdm(range(1, args.iterations + 1))
    for iteration in progress_bar:
        progress_bar.set_description(desc=f'Iter {iteration}')
        

        if iteration in args.upsample_iterations:
            # Upsample the mesh by remeshing the surface with half the average edge length
            e0, e1 = mesh.edges.unbind(1)
            average_edge_length = torch.linalg.norm(mesh.vertices[e0] - mesh.vertices[e1], dim=-1).mean()
            v_upsampled, f_upsampled = remesh_botsch(mesh.vertices.cpu().detach().numpy().astype(np.float64), mesh.indices.cpu().numpy().astype(np.int32), h=float(average_edge_length/2))
            v_upsampled = np.ascontiguousarray(v_upsampled)
            f_upsampled = np.ascontiguousarray(f_upsampled)

            mesh_initial = Mesh(v_upsampled, f_upsampled, device=device)
            mesh_initial.compute_connectivity()

            # Adjust weights and step size
            loss_weights['laplacian'] *= 4
            #loss_weights['normal'] *= 4
            loss_weights['normal_c'] *= 4
            lr_vertices *= 0.75

            # Create a new optimizer for the vertex offsets
            vertex_offsets = torch.zeros_like(mesh_initial.vertices)
            vertex_offsets.requires_grad = True
            optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)

        # Deform the initial mesh
        mesh = mesh_initial.with_vertices(mesh_initial.vertices + vertex_offsets)

        # Sample a view subset
        views_subset = view_sampler(views)

        # Render the mesh from the views
        # Perform antialiasing here because we cannot antialias after shading if we only shade a some of the pixels
        gbuffers = renderer.render(views_subset, mesh, channels=['mask', 'position', 'normal', 'depth'], with_antialiasing=True) 

        # Combine losses and weights
        if loss_weights['mask'] > 0:
            losses['mask'] = mask_loss(views_subset, gbuffers)
        if loss_weights['normal'] > 0:
            if args.loss == "L2":
                losses['normal'] = normal_loss(views_subset, gbuffers, torch.nn.MSELoss(), args.compare_size, device=device)
            else:
                losses['normal'] = normal_loss(views_subset, gbuffers, torch.nn.L1Loss(), args.compare_size, device=device)
        if loss_weights['depth'] > 0:
            losses['depth'] = depth_loss(views_subset, gbuffers, args.compare_size, device=device)
        if loss_weights['normal_c'] > 0:
            losses['normal_c'] = normal_consistency_loss(mesh)
        if loss_weights['laplacian'] > 0:
            losses['laplacian'] = laplacian_loss(mesh)
        if loss_weights['shading'] > 0:
            losses['shading'] = shading_loss(views_subset, gbuffers, shader=shader, shading_percentage=args.shading_percentage)

        loss = torch.tensor(0., device=device)
        for k, v in losses.items():
            loss += v * loss_weights[k]

        # Optimize
        optimizer_vertices.zero_grad()
        optimizer_shader.zero_grad()
        loss.backward()
        optimizer_vertices.step()
        optimizer_shader.step()

        #----------------------------------------Visualizing GPU usage and losses ------------------------------------

        # Compute GPU memory stats
        current = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        remaining = total - current

        # Set GPU memory stats
        progress_bar.set_postfix(
            {
            'loss': f'{loss.detach().cpu():.2f}',
            'n_loss': f'{losses["normal"].detach().cpu()*loss_weights["normal"]:.2f}', 
            'd_loss': f'{losses["depth"].detach().cpu()*loss_weights["depth"]:.2f}',
            'Curr': f'{current / 1024**2:.0f}MB', 
            'Peak': f'{peak / 1024**2:.0f}MB', 
            'Tot': f'{total / 1024**2:.0f}MB'
            }, refresh=True)


        #progress_bar.set_postfix({'loss': loss.detach().cpu(), 'normal_loss':losses['normal'].detach().cpu()*loss_weights['normal']})


        # Visualizations
        if (args.visualization_frequency > 0) and shader and (iteration == 1 or iteration % args.visualization_frequency == 0):
            
            with torch.no_grad():
                use_fixed_views = len(args.visualization_views) > 0
                view_indices = args.visualization_views if use_fixed_views else [np.random.choice(list(range(len(views_subset))))]
                for vi in view_indices:
                    debug_view = views[vi] if use_fixed_views else views_subset[vi]
                    debug_gbuffer = renderer.render([debug_view], mesh, channels=['mask', 'position', 'normal', 'depth'], with_antialiasing=True)[0]
                    position = debug_gbuffer["position"]
                    normal = debug_gbuffer["normal"]
                    depth = debug_gbuffer["depth"].cpu()

                    view_direction = torch.nn.functional.normalize(debug_view.camera.center - position, dim=-1)

                    # Save the shaded rendering
                    shaded_image = shader(position, normal, view_direction) * debug_gbuffer["mask"] + (1-debug_gbuffer["mask"])
                    shaded_path = (images_save_path / str(vi) / "shaded") if use_fixed_views else (images_save_path / "shaded")
                    shaded_path.mkdir(parents=True, exist_ok=True)
                    plt.imsave(shaded_path / f'neuralshading_{iteration}.png', shaded_image.cpu().numpy())

                    # Save a normal map in camera space
                    #normal = torch.clamp(normal, min=0)
                    normal_path = (images_save_path / str(vi) / "normal") if use_fixed_views else (images_save_path / "normal")
                    normal_path.mkdir(parents=True, exist_ok=True)
                    R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)
                    normal_image = (0.5*(normal @ debug_view.camera.R.T + 1)) * debug_gbuffer["mask"] + (1-debug_gbuffer["mask"])
                    #normal_image *= 255
                    #normal_image = (normal)*255
                    #normal_image = normal_image * debug_gbuffer["mask"].expand_as(normal_image)
                    #normal_image = normal_image.cpu().numpy().astype(np.uint8)
                    #plt.imsave(normal_path / f'neuralshading_{iteration}.png', normal_image)
                    #normal_image = (0.5*(normal @ debug_view.camera.R.T + 1)) * debug_gbuffer["mask"] + (1-debug_gbuffer["mask"])


                    #normal_image = (0.5*(normal + 1)*255)
                    #normal_image = normal_image * debug_gbuffer["mask"].expand_as(normal_image)
                    #normal_image = normal_image.cpu().numpy().astype(np.uint8)
                    plt.imsave(normal_path / f'neuralshading_{iteration}.png', normal_image.cpu().numpy())

                    # Save a depth map in camera space
                    depth_path = (images_save_path / str(vi) / "depth") if use_fixed_views else (images_save_path / "depth")
                    depth_path.mkdir(parents=True, exist_ok=True)
                    depth_image = depth.squeeze()
                    plt.imsave(depth_path / f'neuralshading_{iteration}.png', depth_image.cpu().numpy(), cmap='gray')

        if (args.save_frequency > 0) and (iteration == 1 or iteration % args.save_frequency == 0):
            with torch.no_grad():
                mesh_for_writing = space_normalization.denormalize_mesh(mesh.detach().to('cpu'))
                write_mesh(meshes_save_path / f"mesh_{iteration:06d}.obj", mesh_for_writing)                                
            shader.save(shaders_save_path / f'shader_{iteration:06d}.pt')

            try:
                if loss_weights['mask'] > 0:
                    losses_record['mask'].append(losses['mask'].cpu().item())
                else:
                    losses_record['mask'].append(0)

                if loss_weights['normal'] > 0:
                    losses_record['normal'].append(losses['normal'].cpu().item())
                else:
                    losses_record['normal'].append(0)    

                if loss_weights['depth'] > 0:
                    losses_record['depth'].append(losses['depth'].cpu().item())
                else:
                    losses_record['depth'].append(0)

                if loss_weights['normal_c'] > 0:
                    losses_record['normal_c'].append(losses['normal_c'].cpu().item())
                else:
                    losses_record['normal_c'].append(0)

                if loss_weights['laplacian'] > 0:
                    losses_record['laplacian'].append(losses['laplacian'].cpu().item())
                else:
                    losses_record['laplacian'].append(0)

                if loss_weights['shading'] > 0:
                    losses_record['shading'].append(losses['shading'].cpu().item())
                else:
                    losses_record['shading'].append(0)                    
            except:
                print('There was an error while saving the losses')
    
    save_losses(losses_record, plots_save_path)

    mesh_for_writing = space_normalization.denormalize_mesh(mesh.detach().to('cpu'))
    write_mesh(meshes_save_path / f"mesh_{args.iterations:06d}.obj", mesh_for_writing)

    if shader is not None:
        shader.save(shaders_save_path / f'shader_{args.iterations:06d}.pt')