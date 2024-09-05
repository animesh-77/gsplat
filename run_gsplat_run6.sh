#!/bin/bash
{
# Exit immediately if a command exits with a non-zero status
set -e


# check number of arguments is 3
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "Usage: run_gsplat_run6.sh <dress_name> <data_factor> <run_grounding_mask=false>"
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
sparse_folder="/cs/student/projects4/ml/2023/asrivast/datasets/$1/dense_png"
output_folder="/cs/student/projects4/ml/2023/asrivast/datasets/$1/$result_dir"
# output_folder="/cs/student/projects4/ml/2023/asrivast/datasets/dress2_3/upright/results_debug"

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"
cat <<EOL >> "$output_folder/details.txt"

run 6 loss: ssim+L1, no mask, test_every 8, data_factor $1, ssim_lambda 0.2 , sh_degree 3, steps_scaler 1 PNG
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
echo "RESULTS AT $output_folder/run_6"
# read -p "Press any key to continue..."
sleep 5

################## run 6 ##################
echo "Running run 6"
python examples/simple_trainer.py \
    --data_dir $sparse_folder \
    --data_factor "$2" \
    --result_dir $output_folder \
    --run 6 \
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

}