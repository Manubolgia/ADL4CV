#!/bin/bash

input_dir_values=("/mnt/hdd/data/omni_skull/views_ocv" "/mnt/hdd/data/omni_hotdog/views_ocv" "/mnt/hdd/data/omni_owl/views_ocv" "/mnt/hdd/data/omni_chair/views_ocv")
iterations_values=(10 100 500 1000 1500 2000)
fixed_iterations=2000
num_views_values_1=(5 25 50 90 -1)#nerf
num_views_values_2=(5 10 20 30 -1)
fixed_num_views=-1
vis_freq=250

weight_normal_values=(0 0.001) # replace 'value1' with your own value
weight_depth_values=(0 0.01) # replace 'value2' with your own value



mkdir -p ./out/iterations_change
mkdir -p ./out/views_change

# Changing number of iterations
current_experiment=0
total_experiments=$((${#input_dir_values[@]} * ${#iterations_values[@]} * 4 + (16*5)))
for input_dir in ${input_dir_values[@]}
do
    input_bbox="${input_dir%/views_ocv}/bbox.txt"

    for iterations in ${iterations_values[@]}
    do
        for weight_normal in ${weight_normal_values[@]}
        do
            for weight_depth in ${weight_depth_values[@]}
            do
                current_experiment=$((current_experiment+1))
                dataset_name=$(basename $(dirname $input_dir))
                run_name="${dataset_name}_it${iterations}_v${fixed_num_views}_wN${weight_normal//./-}_wD${weight_depth//./-}"

                echo "Starting experiment $current_experiment of $total_experiments with: $run_name"

                python3 reconstruct.py \
                    --input_dir $input_dir \
                    --input_bbox $input_bbox \
                    --run_name $run_name \
                    --iterations $iterations \
                    --num_views $fixed_num_views \
                    --weight_normal $weight_normal \
                    --weight_depth $weight_depth \
                    --shade_views_eval True \
                    --output_dir ./out/iterations_change \
                    --visualization_frequency $vis_freq > ./out/iterations_change/${run_name}.log

                echo "Finished experiment $current_experiment of $total_experiments with: $run_name"
            done
        done
    done

    if [ "$input_dir" = "/mnt/hdd/data/omni_hotdog/views_ocv" ] || [ "$input_dir" = "/mnt/hdd/data/omni_chair/views_ocv" ]; then
        num_views_values=("${num_views_values_1[@]}")
    else
        num_views_values=("${num_views_values_2[@]}")
    fi

    for num_views in ${num_views_values[@]}
    do
        for weight_normal in ${weight_normal_values[@]}
        do
            for weight_depth in ${weight_depth_values[@]}
            do
                current_experiment=$((current_experiment+1))
                dataset_name=$(basename $(dirname $input_dir))
                run_name="${dataset_name}_it${fixed_iterations}_v${num_views}_wN${weight_normal//./-}_wD${weight_depth//./-}"

                echo "Starting experiment $current_experiment of $total_experiments with: $run_name"

                python3 reconstruct.py \
                    --input_dir $input_dir \
                    --input_bbox $input_bbox \
                    --run_name $run_name \
                    --iterations $fixed_iterations \
                    --num_views $num_views \
                    --weight_normal $weight_normal \
                    --weight_depth $weight_depth \
                    --shade_views_eval True \
                    --output_dir ./out/views_change \
                    --visualization_frequency $vis_freq > ./out/views_change/${run_name}.log

                echo "Finished experiment $current_experiment of $total_experiments with: $run_name"
            done
        done
    done
done
