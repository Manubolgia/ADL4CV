#!/bin/bash

# Base directory for experiments
base_dir="/home/pavkovic/ADL4CV"

# Directory for output csv files
csv_out="/home/obelleiro/ADL4CV/csv_out"

# Create the output directory if it doesn't exist
mkdir -p $csv_out

# Define the datasets and their types
datasets=("omni_hotdog" "omni_chair" "omni_skull" "omni_owl")
dataset_types=("nerf" "nerf" "dtu" "dtu")

# Define the models
models=("_wN0_wD0" "_wN0-001_wD0" "_wN0_wD0-01" "_wN0-001_wD0-01")

# Define the iterations and views for each dataset type
nerf_iterations=(10 100 500 1000 1500 2000)
nerf_views=(5 25 50 90 -1)
dtu_iterations=(10 100 500 1000 1500 2000)
dtu_views=(5 10 20 30 -1)

# Iterate over datasets
for i in ${!datasets[@]}; do
  # Choose the correct set of iterations and views based on dataset type
  if [ ${dataset_types[$i]} = "nerf" ]; then
    iterations=("${nerf_iterations[@]}")
    views=("${nerf_views[@]}")
  else
    iterations=("${dtu_iterations[@]}")
    views=("${dtu_views[@]}")
  fi

  # Create CSV files for changing iterations (views fixed at -1)
  output_file="${csv_out}/${datasets[$i]}_iterations.csv"
  echo "iteration,fscore_${models[0]},loss_${models[0]},psnr_${models[0]},fscore_${models[1]},loss_${models[1]},psnr_${models[1]},fscore_${models[2]},loss_${models[2]},psnr_${models[2]},fscore_${models[3]},loss_${models[3]},psnr_${models[3]}" > $output_file
  for it in ${iterations[@]}; do
    row="$it"
    for model in ${models[@]}; do
      # Adjust the file paths and contents based on the dataset type
      if [ ${dataset_types[$i]} = "nerf" ]; then
        mesh_eval_file="${base_dir}/eval_iterations/${datasets[$i]}_it${it}_v-1${model}_mesh_eval.txt"
        fscore=$(grep "fscore:" $mesh_eval_file | awk '{print $2}')
        loss=$(grep "loss:" $mesh_eval_file | awk '{print $2}')
      else
        mesh_eval_file="${base_dir}/eval_iterations/${datasets[$i]}_it${it}_v-1${model}.txt"
        fscore="NA"
        loss=$(grep "overall:" $mesh_eval_file | awk '{print $2}')
      fi
      shading_eval_file="${base_dir}/eval_iterations/${datasets[$i]}_it${it}_v-1${model}_${dataset_types[$i]}_shading_eval.txt"
      psnr=$(grep "psnr score:" $shading_eval_file | awk '{print $3}')

      row="$row,$fscore,$loss,$psnr"
    done
    echo $row >> $output_file
  done

  # Create CSV files for changing views (iterations fixed at 2000)
  output_file="${csv_out}/${datasets[$i]}_views.csv"
  echo "view,fscore_${models[0]},loss_${models[0]},psnr_${models[0]},fscore_${models[1]},loss_${models[1]},psnr_${models[1]},fscore_${models[2]},loss_${models[2]},psnr_${models[2]},fscore_${models[3]},loss_${models[3]},psnr_${models[3]}" > $output_file
  for view in ${views[@]}; do
    row="$view"
    for model in ${models[@]}; do
      # Adjust the file paths and contents based on the dataset type
      if [ ${dataset_types[$i]} = "nerf" ]; then
        mesh_eval_file="${base_dir}/eval_views/${datasets[$i]}_it2000_v${view}${model}_mesh_eval.txt"
        fscore=$(grep "fscore:" $mesh_eval_file | awk '{print $2}')
        loss=$(grep "loss:" $mesh_eval_file | awk '{print $2}')
      else
        mesh_eval_file="${base_dir}/eval_views/${datasets[$i]}_it2000_v${view}${model}.txt"
        fscore="NA"
        loss=$(grep "overall:" $mesh_eval_file | awk '{print $2}')
      fi
      shading_eval_file="${base_dir}/eval_views/${datasets[$i]}_it2000_v${view}${model}_${dataset_types[$i]}_shading_eval.txt"
      psnr=$(grep "psnr score:" $shading_eval_file | awk '{print $3}')

      row="$row,$fscore,$loss,$psnr"
    done
    echo $row >> $output_file
  done
done
