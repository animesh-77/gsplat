#!/bin/bash
{
# Exit immediately if a command exits with a non-zero status
set -e


# check number of arguments is 3
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "Usage: run_gsplat_run2.sh <dress_name> <data_factor> <run_grounding_mask=false>"
    exit 1
fi


dress_name=$1
data_factor=$2

if [ "$data_factor" = "8" ]; then
    result_dir="results_1"
    image_dir="images_8"
elif [ "$data_factor" = "4" ]; then
    result_dir="results_2"
    image_dir="images_4"
elif [ "$data_factor" = "2" ]; then
    result_dir="results_3"
    image_dir="images_2"
else
    result_dir="results_4"
    image_dir="images"
fi
# Activate the desired conda environment
# conda activate gaussian_splatting
# Run gsplat on a list of input files
# first define some arguments
sparse_folder="/cs/student/projects4/ml/2023/asrivast/datasets/$1/dense"
output_folder="/cs/student/projects4/ml/2023/asrivast/datasets/$1/$result_dir"
# output_folder="/cs/student/projects4/ml/2023/asrivast/datasets/dress2_3/upright/results_debug"

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"
cat <<EOL >> "$output_folder/details.txt"

run 0 loss: ssim+L1, no mask, test_every 8, data_factor $1, ssim_lambda 0.2 , sh_degree 3, steps_scaler 1
EOL


# check if first agrument passed in true or false
if [ "$3" = true ]; then
    ################## Run Detectron2_SAM ##################
    # python examples/detectron2_SAM_bbox.py \
    #         --image_path $sparse_folder/images

    ################## Run DINO_SAM.py ##################
    python examples/Grounding_SAM.py \
            --image_path $sparse_folder/$image_dir
    # read -p "Check the masks and press any key to continue with 3DGS..."
    # exit 0
else 
    echo "Skipping the GROUNDING SAM step"
fi
echo "SPARSE FOLDER $sparse_folder"
echo "RESULTS AT $output_folder/run 0"
# read -p "Press any key to continue..."
sleep 5

################## run 0 ##################
echo "Running run 0"
python examples/simple_trainer.py \
    --data_dir $sparse_folder \
    --data_factor "$2" \
    --result_dir $output_folder \
    --run 0 \
    --test_every 8 \
    --train_cam_ids 0 \
    --test_cam_ids 0 \
    --max_steps 30000 \
    --save_steps 1 9000 21000 \
    --refine_start_iter 100 \
    --refine_stop_iter 15000 \
    --reset_every 500 \
    --sh_degree_interval 900 \
    --sh_degree 3 \
    --refine_every 100 \
    --ssim_lambda 0.2 \
    --steps_scaler 1 \
    --disable_viewer #\
    # --masked 
################## Run 0 ##################



################## Run 1 ##################
# echo "Running run 1"
# Path to the JSON file from previous run
# json_file=$(find $output_folder/$run/config_files -name "*.json" -type f -print -quit)
# next_run=$output_folder/$run/next_run
# # Create the directory if it doesn't exist
# mkdir -p "$next_run"
# train_cam_ids="[0, 8, 12, 21, 25, 33, 59, 44, 48, 57]"
# # jq --argjson train_cam_ids "$train_cam_ids" 'del(.train_images, .test_images) | .train_cam_ids = $train_cam_ids | .run = 1 | .test_every = null | .refine_stop_iter = -1 | .masked = false' "$json_file" > "$next_run/$(basename "$json_file")"

# jq --argjson train_cam_ids "$train_cam_ids" 'del(.train_images, .test_images) | .run = 1 | .refine_stop_iter = -1 ' "$json_file" > "$next_run/$(basename "$json_file")"
# echo "Using JSON file: $next_run/$(basename "$json_file")"
# # Path to ckpt file from previous run
# ckpt_file=$(find $output_folder/$run/ckpts -name "ckpt_*.pt" -type f | sort -r -V | head -n 1)
# echo "Using ckpt file: $ckpt_file"
# # read -p "Press any key to continue..."
# python examples/simple_trainer.py \
#     --ckpt $ckpt_file \
#     --resume \
#     # --disable_viewer 
# run="run1"
# ################## Run 1 ##################

# ################## Run 2 ##################
# echo "Running run 2"
# Path to the JSON file from previous run
# json_file=$(find $output_folder/$run/config_files -name "*.json" -type f -print -quit)
# next_run=$output_folder/$run/next_run
# # Create the directory if it doesn't exist
# mkdir -p "$next_run"
# # change values in the JSON file
# train_cam_ids="[0, 7, 14, 21, 28, 35, 42, 49, 56]"
# # jq ' del(.train_images, .test_images) | .run = 2 | .refine_stop_iter = -1 | .masked = false ' "$json_file" > "$next_run/$(basename "$json_file")"
# jq --argjson train_cam_ids "$train_cam_ids" 'del(.train_images, .test_images) | .run = 2 | .refine_stop_iter = -1 | .ssim_lambda = 0.6' "$json_file" > "$next_run/$(basename "$json_file")"
# echo "Using JSON file: $next_run/$(basename "$json_file")"
# # Path to ckpt file from previous run the lastest ckpt file
# ckpt_file=$(find $output_folder/$run/ckpts -name "ckpt_*.pt" -type f | sort -r -V | head -n 1)
# echo "Using ckpt file: $ckpt_file"
# # read -p "Press any key to continue..."
# python examples/simple_trainer.py \
#     --ckpt $ckpt_file \
#     --resume \
#     --disable_viewer 
# run="run2"
# ################## Run 2 ##################
}