#!/bin/bash

weight_normal_c=0.1
compare_size=200
loss='L1'
vis_freq=500

weight_depth_values=(0.001 0.005 0.01 0.05)
weight_normal_values=(0.001 0.005 0.01 0.05)
normal_weight_factor_values=(0.9 1 1.5)
depth_weight_factor_values=(0.9 1 1.5)

for weight_depth in ${weight_depth_values[@]}
do
    for weight_normal in ${weight_normal_values[@]}
    do
        for normal_weight_factor in ${normal_weight_factor_values[@]}
        do
            for depth_weight_factor in ${depth_weight_factor_values[@]}
            do
                run_name="omni_hotdog_wD${weight_depth}_wN${weight_normal}_${loss}_nWF${normal_weight_factor}_dWF${depth_weight_factor}"

                echo "Running with: $run_name"

                python3 reconstruct.py \
                    --input_dir /mnt/hdd/data/omni_hotdog/views_ocv \
                    --input_bbox /mnt/hdd/data/omni_hotdog/bbox.txt \
                    --run_name $run_name \
                    --weight_normal_c $weight_normal_c \
                    --weight_depth $weight_depth \
                    --weight_normal $weight_normal \
                    --loss $loss \
                    --compare_size $compare_size \
                    --normal_weight_factor $normal_weight_factor \
                    --depth_weight_factor $depth_weight_factor \
                    --visualization_frequency $vis_freq > /home/obelleiro/ADL4CV/out/log_files/${run_name}.log

                echo "Finished running with: $run_name"
            done
        done
    done
done

# Special run with L2 loss
loss='L2'
weight_depth=0.01
weight_normal=0.01
normal_weight_factor=1
depth_weight_factor=1
run_name="omni_hotdog_special_L2"

echo "Running special L2 run: $run_name"

python3 reconstruct.py \
    --input_dir /mnt/hdd/data/omni_hotdog/views_ocv \
    --input_bbox /mnt/hdd/data/omni_hotdog/bbox.txt \
    --run_name $run_name \
    --weight_normal_c $weight_normal_c \
    --weight_depth $weight_depth \
    --weight_normal $weight_normal \
    --loss $loss \
    --compare_size $compare_size \
    --normal_weight_factor $normal_weight_factor \
    --depth_weight_factor $depth_weight_factor \
    --visualization_frequency $vis_freq > /home/obelleiro/ADL4CV/out/log_files/${run_name}.log

echo "Finished special L2 run: $run_name"

# Special run with weight_normal_c=1.2
weight_normal_c=1.2
loss='L1'
weight_depth=0.01
weight_normal=0.01
normal_weight_factor=1
depth_weight_factor=1
run_name="omni_hotdog_special_wNC1.2"

echo "Running special weight_normal_c=1.2 run: $run_name"

python3 reconstruct.py \
    --input_dir /mnt/hdd/data/omni_hotdog/views_ocv \
    --input_bbox /mnt/hdd/data/omni_hotdog/bbox.txt \
    --run_name $run_name \
    --weight_normal_c $weight_normal_c \
    --weight_depth $weight_depth \
    --weight_normal $weight_normal \
    --loss $loss \
    --compare_size $compare_size \
    --normal_weight_factor $normal_weight_factor \
    --depth_weight_factor $depth_weight_factor \
    --visualization_frequency $vis_freq > /home/obelleiro/ADL4CV/out/log_files/${run_name}.log

echo "Finished special weight_normal_c=1.2 run: $run_name"

# Special run with compare_size=400
weight_normal_c=0.1
compare_size=400
run_name="omni_hotdog_special_CS400"

echo "Running special compare_size=400 run: $run_name"

python3 reconstruct.py \
    --input_dir /mnt/hdd/data/omni_hotdog/views_ocv \
    --input_bbox /mnt/hdd/data/omni_hotdog/bbox.txt \
    --run_name $run_name \
    --weight_normal_c $weight_normal_c \
    --weight_depth $weight_depth \
    --weight_normal $weight_normal \
    --loss $loss \
    --compare_size $compare_size \
    --normal_weight_factor $normal_weight_factor \
    --depth_weight_factor $depth_weight_factor \
    --visualization_frequency $vis_freq > /home/obelleiro/ADL4CV/out/log_files/${run_name}.log

echo "Finished special compare_size=400 run: $run_name"
