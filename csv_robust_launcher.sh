#!/bin/bash

# Directory settings
base_dir="/home/pavkovic/ADL4CV/eval_robustness"
csv_out="/home/obelleiro/ADL4CV/csv_out"
mkdir -p $csv_out

# Only dataset
dataset="omni_chair"

# List of models
models=("_wN0_wD0" "_wN0-001_wD0" "_wN0_wD0-01" "_wN0-001_wD0-01")

# Prepare header for CSV file
header="run"
for model in ${models[@]}; do
    clean_model_name=$(echo $model | tr -d "_")
    header="$header,fscore_$clean_model_name,loss_$clean_model_name,psnr_dtu_$clean_model_name,psnr_nerf_$clean_model_name"
done

# Create CSV file
output_file="${csv_out}/${dataset}.csv"
echo $header > $output_file

# For each number from 0 to 9
for number in {0..9}; do
    row="$number"
    # For each model
    for model in ${models[@]}; do
        fscore="NaN"
        loss="NaN"
        psnr_dtu="NaN"
        psnr_nerf="NaN"
        # File path for mesh_eval
        mesh_eval_file="${base_dir}/${dataset}_${number}${model}_mesh_eval.txt"
        # If file exists, extract the fscore and loss
        if [ -f "$mesh_eval_file" ]; then
            fscore=$(grep "fscore:" $mesh_eval_file | awk '{print $2}')
            loss=$(grep "loss:" $mesh_eval_file | awk '{print $2}')
        fi
        # File path for dtu_shading_eval
        dtu_shading_eval_file="${base_dir}/${dataset}_${number}${model}_dtu_shading_eval.txt"
        # If file exists, extract the psnr
        if [ -f "$dtu_shading_eval_file" ]; then
            psnr_dtu=$(grep "psnr score:" $dtu_shading_eval_file | awk '{print $3}')
        fi
        # File path for nerf_shading_eval
        nerf_shading_eval_file="${base_dir}/${dataset}_${number}${model}_nerf_shading_eval.txt"
        # If file exists, extract the psnr
        if [ -f "$nerf_shading_eval_file" ]; then
            psnr_nerf=$(grep "psnr score:" $nerf_shading_eval_file | awk '{print $3}')
        fi
        row="$row,$fscore,$loss,$psnr_dtu,$psnr_nerf"
    done
    echo $row >> $output_file
done
