#!/bin/bash

while getopts i:v:d: flag
do
    case "${flag}" in
        i) input_path=${OPTARG};;
        v) views=${OPTARG};;
        d) dataset=${OPTARG};;
    esac
done

input_dir="${input_path}/${views}"
cd "${input_dir}/views"
imgs_num=$(ls | wc -l)
let "imgs_num = $imgs_num / 5"
quarter=$((imgs_num / 4))
half=$((imgs_num / 2))
cd ~/ADL4CV
for num in 12 #10 $quarter $half $imgs_num
do
        ./reconstruct.sh -i $input_path -v $views -n $num
        
        if [ $dataset = "DTU" ]
        then
                cd DTUeval-python/
                scan_id=$(echo $views | cut -d "_" -f 1)
                python eval.py --data ~/ADL4CV/out/${views}_${num}_std/meshes/mesh_006000.obj --scan $scan_id --mode mesh --dataset_dir ~/DTU/Offical_DTU_Dataset/ --vis_out_dir eval_results/
                python eval.py --data ~/ADL4CV/out/${views}_${num}_norm/meshes/mesh_006000.obj --scan $scan_id --mode mesh --dataset_dir ~/DTU/Offical_DTU_Dataset/ --vis_out_dir eval_results/
                cd ..
        fi
        if [ $dataset = "NeRF" ]
        then
                #for i in 1000 3000 6000
                #do
                        cd NeRFeval/
                        python eval.py --mesh ~/ADL4CV/out/${views}_${num}_std/meshes --ref ~/NERF_Dataset_ocv/ground_truth/${views}_gt.obj --iter 2000 -n 2500000
                        python eval.py --mesh ~/ADL4CV/out/${views}_${num}_norm_l1/meshes --ref ~/NERF_Dataset_ocv/ground_truth/${views}_gt.obj --iter 2000 -n 2500000
                        python eval.py --mesh ~/ADL4CV/out/${views}_${num}_norm_l2/meshes --ref ~/NERF_Dataset_ocv/ground_truth/${views}_gt.obj --iter 2000 -n 2500000
                        cd ..
                #done
        fi
done
