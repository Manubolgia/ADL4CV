# DTUeval-python

A python implementation of DTU MVS 2014 evaluation. It only takes 1min for each mesh evaluation. And the gap between the two implementations is negligible. 

## Setup and Usage

This script requires the following dependencies.

```
numpy open3d scikit-learn tqdm scipy multiprocessing argparse
```

Download the STL point clouds and Sample Set and prepare the ground truth folder as follows.

```
<dataset_dir>
- Points
    - stl
        - stlxxx_total.ply
- ObsMask
    - ObsMaskxxx_10.mat
    - Planexxx.mat
```

Run the evaluation script (e.g. scan24, mesh mode)
```
python eval.py --data <input> --scan 24 --mode mesh --dataset_dir <dataset_dir> --vis_out_dir <out_dir_for_visualization> --iter <number_of_reconstruction_iterations>
```
where input is the folder containing reconstructed mesh and number_of_reconstruction_iterations number in the mesh name

## Discussion on randomness
There is randomness in point cloud downsampling in both versions. It iterates through the points and delete the points with distance < 0.2. So the order of points matters. We randomly shuffle the points before downsampling. 

## Error visualization
`vis_xxx_d2s.ply` and `vis_xxx_s2d.ply` are error visualizations.
- Blue: Out of bounding box or ObsMask
- Green: Errors larger than threshold (20)
- White to Red: Errors counted in the reported statistics