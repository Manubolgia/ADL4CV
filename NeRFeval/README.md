### Requirements
The requirements are PyTotch and Pytorch3D with cuda support:
  * [PyTorch>=1.1.0](https://pytorch.org/get-started/locally/) 
  * [PyTorch3D with CUDA support](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) 

### Installation
1. Install PyTorch (>= 1.1.0)
2. Install PyTorch3d gpu version.
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
3. To install the package simply run the following line:
```
pip install git+'https://github.com/otaheri/chamfer_distance'

```
4. Install open3d

### Usage
Run the evaluation script
```
python eval_shading.py --input_dir <input_views> --input_bbox <bbox> --mesh <mesh_folder> --shader <shader_folder> --iter <number_of_reconstruction_iterations> --output_dir <output_directory>
```
where mesh_folder is the folder containing reconstructed mesh, where shader_folder is the folder containing trained shader, input_views folder containing input views, bbox is input bounding box file and number_of_reconstruction_iterations number in the mesh name

