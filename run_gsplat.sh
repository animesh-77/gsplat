#!/bin/bash

# Run gsplat on a list of input files
# first define some arguments
sparse_folder="/cs/student/projects4/ml/2023/asrivast/datasets/dress3_15/upright/sparse/undistorted"
output_folder="/cs/student/projects4/ml/2023/asrivast/datasets/dress3_15/upright/results"
run="run0"
# Create .tmp directory if it doesn't exist
mkdir -p .tmp
# run 0
echo "Running run 0"
python examples/simple_trainer.py \
    --data_dir $sparse_folder \
    --data_factor 1 \
    --result_dir $output_folder \
    --train_cam_ids 1 50 \
    --test_cam_ids 2 15 45 \
    --max_steps 3000 \
    --save_steps 800 1800 \



# Path to the JSON file from previous run
json_file=$(find $output_folder/$run/config_files -name "*.json" -type f -print -quit)
next_run=$output_folder/$run/next_run
# Create the directory if it doesn't exist
mkdir -p "$next_run"

echo "Using JSON file: $json_file"
train_cam_ids="[7, 17]"
jq 'del(.train_images, .test_images) | .train_cam_ids = $train_cam_ids | .run = 1' --argjson train_cam_ids "$train_cam_ids" "$json_file" \
    > "$next_run/$(basename "$json_file")"

# Path to ckpt file from previous run
ckpt_file=$(find $output_folder/$run/ckpts -name "ckpt_*.pt" -type f | sort -r -V | head -n 1)
echo "Using ckpt file: $ckpt_file"

# read -p "Press any key to continue..."
# run 1

echo "Running run 1"
python examples/simple_trainer.py \
    --ckpt $ckpt_file \
    --resume \
