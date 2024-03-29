#!/bin/bash

input_dir_values=("/mnt/hdd/data/omni_chair/views_ocv")
vis_freq=250

weight_normal_values=(0 0.001) 
weight_depth_values=(0 0.01)

iterations=2000

mkdir -p ./out/result_robustness

# Changing number of iterations
current_experiment=0
max_experiments=10
total_experiments=10
for input_dir in ${input_dir_values[@]}
do
    input_bbox="${input_dir%/views_ocv}/bbox.txt"
    while [ $current_experiment -lt $max_experiments ]
    do
        for weight_normal in ${weight_normal_values[@]}
        do
            for weight_depth in ${weight_depth_values[@]}
            do
                dataset_name=$(basename $(dirname $input_dir))
                run_name="${dataset_name}_${current_experiment}_wN${weight_normal//./-}_wD${weight_depth//./-}"

                echo "Starting experiment $current_experiment of $total_experiments with: $run_name"

                python3 reconstruct.py \
                    --input_dir $input_dir \
                    --input_bbox $input_bbox \
                    --run_name $run_name \
                    --iterations $iterations \
                    --weight_normal $weight_normal \
                    --weight_depth $weight_depth \
                    --shade_views_eval True \
                    --output_dir ./out/result_robustness \
                    --visualization_frequency $vis_freq > ./out/result_robustness/${run_name}.log

                echo "Finished experiment $current_experiment of $total_experiments with: $run_name"
            done
        done
        current_experiment=$((current_experiment+1))
    done
done
