#!/bin/bash

while getopts i:v: flag
do
    case "${flag}" in
        i) input_path=${OPTARG};;
        v) views=${OPTARG};;
    esac
done

for dir in $input_path/*/
do
    if [[ $dir == *"temp/" ]]
    then
        echo "logs files"
    else
        echo $dir
        python3 ~/ADL4CV/NeRFeval/eval_shading.py --input_dir $views --input_bbox ${dir}bbox.txt --mesh ${dir}meshes --shader ${dir}shaders/shader_002000.pt --iter 2000 --output_dir $dir
        python ~/ADL4CV/NeRFeval/eval.py --mesh ${dir}meshes --ref ~/NERF_Dataset_ocv/ground_truth/hotdog_gt.obj --iter 2000 -n 2500000 --out $dir
    fi
done