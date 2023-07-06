#!/bin/bash

# Directory settings
base_dir="/home/pavkovic/ADL4CV/eval_robustness"
csv_out="/home/pavkovic/ADL4CV/csv_out"
mkdir -p $csv_out

# Only dataset
dataset="omni_chair"

# List of models
models=("_wN0_wD0" "_wN0-001_wD0" "_wN0_wD0-01" "_wN0-001_wD0-01")

# List of metrics
metrics=("dtu_shading_eval" "mesh_eval")

# Create CSV file
output_file="${csv_out}/${dataset}.csv"
echo "run,fscore_model1,loss_model1,psnr_model1,fscore_model2,loss_model2,psnr_model2,fscore_model3,loss_model3,psnr_model3,fscore_model4,loss_model4,psnr_model4" > $output_file

# For each number from 0 to 9
for number in {0..9}; do
    row="$number"
    # For each model and metric
    for model in ${models[@]}; do
        fscore="NaN"
        loss="NaN"
        psnr="NaN"
        for metric in ${metrics[@]}; do
            # File path
            eval_file="${base_dir}/${dataset}_${number}${model}_${metric}.txt"
            # If file exists, extract the metrics
            if [ -f "$eval_file" ]; then
                fscore=$(grep "fscore:" $eval_file | awk '{print $2}')
                loss=$(grep "loss:" $eval_file | awk '{print $2}')
                psnr=$(grep "psnr score:" $eval_file | awk '{print $3}')
            fi
        done
        row="$row,$fscore,$loss,$psnr"
    done
    echo $row >> $output_file
done
