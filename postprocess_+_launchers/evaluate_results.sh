#!/bin/bash

while getopts i: flag
do
    case "${flag}" in
        i) input_path=${OPTARG};;
    esac
done

for dir in $input_path/*/
do
    if [[ $dir == *"log_files/" ]]
    then
        echo "logs files"
    else
        dataset=$(echo $dir | rev | cut -d "/" -f 2 | rev | cut -d "_" -f 2)
        echo $dataset
        if [ $dataset == "chair" ] || [ $dataset == "hotdog" ] || [ $dataset == "drums" ]
        then
            if [ $dataset == "chair" ]
            then    
                views="/mnt/hdd/data/omni_chair"
            fi
            if [ $dataset == "hotdog" ]
            then    
                views="/mnt/hdd/data/omni_hotdog"
            fi
            if [ $dataset == "drums" ]
            then    
                views="/mnt/hdd/data/omni_drums"
            fi
            number=$(echo "$dir" | grep -oP '(?<=it)\d+')
            echo $dir
            python ~/ADL4CV/NeRFeval/eval_shading.py --input_dir $views/views_ocv --input_bbox ${dir}bbox.txt --mesh ${dir}meshes --shader ${dir}shaders --iter $number --output_dir ~/ADL4CV/eval_robustness
            python ~/ADL4CV/NeRFeval/eval.py --mesh ${dir}meshes --ref ~/NERF_Dataset_ocv/ground_truth/${dataset}_gt.obj --iter $number -n 2500000 --out ~/ADL4CV/eval_robustness
            python ~/ADL4CV/DTUeval-python/eval_shading.py --input_dir $views/views_ocv --input_sh_dir ${dir}images/eval_shaded --input_bbox $views/bbox.txt --output_dir ~/ADL4CV/eval_robustness
        else
            if [ $dataset == "skull" ] || [ $dataset == "owl" ]
            then
                if [ $dataset == "skull" ]
                then    
                    views="/mnt/hdd/data/omni_skull"
                    scan_id=65
                fi
                if [ $dataset == "owl" ]
                then    
                    views="/mnt/hdd/data/omni_owl"
                    scan_id=122
                fi
                number=$(echo "$dir" | grep -oP '(?<=it)\d+')
                echo $dir
                python ~/ADL4CV/DTUeval-python/eval_shading.py --input_dir $views/views_ocv --input_sh_dir ${dir}images/eval_shaded --input_bbox $views/bbox.txt --output_dir ~/ADL4CV/eval_robustness
                #python ~/ADL4CV/NeRFeval/eval_shading.py --input_dir $views --input_bbox ${dir}bbox.txt --mesh ${dir}meshes --shader ${dir}shaders/shader_002000.pt --iter 2000 --output_dir eval_robustness
                python ~/ADL4CV/DTUeval-python/eval.py --data ${dir}meshes --scan $scan_id --mode mesh --dataset_dir ~/DTU/Offical_DTU_Dataset/ --vis_out_dir ~/ADL4CV/eval_robustness --iter $number
            else 
                echo "Not in dataset: $dir"
            fi
        fi
    fi
done