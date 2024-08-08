#!/bin/bash
# Activate the desired conda environment
conda activate gaussian_splatting










# Run gsplat on a list of input files
# first define some arguments
video_dataset="/cs/student/projects4/ml/2023/asrivast/datasets/4DGS/multipleview/portsmouth"
tmp_folder="/cs/student/projects4/ml/2023/asrivast/datasets/4DGS/multipleview/tmp"
output_folder="/cs/student/projects4/ml/2023/asrivast/datasets/4DGS/multipleview/results_1/frame_000001"
mkdir -p $output_folder

# Loop from 0 to 1000
for ((i=7; i<110; i++)); do
    # Get the variable with leading zeros and prefix 'frame_'
    frame_name=$(printf "frame_%06d" $i)




    # Remove the images folder made in previous step
    rm -rf $tmp_folder/images
    rm -rf $tmp_folder/masked_images
    rm -rf $tmp_folder/masks_jpg
    # read -n1 -r -p "$frame_name Delted old folders. Press any key to continue..." key


    # Make the new images folder again all files in this will have the name cam_name_000001.jpg
    mkdir -p $tmp_folder/images
    # Loop from cameras 1 to 32
    for ((j=1; j<=32; j++)); do
        # Get the variable with leading zeros and prefix 'cam'
        cam_name=$(printf "cam%02d" $j)
        # move the .jpg file to tmp folder with the name cam_name_000001.jpg to be consistent with the COLMAP
        cp "$video_dataset/$cam_name/$frame_name.jpg" "$tmp_folder/images/${cam_name}_frame_000001.jpg"
        # break
    done

    # make the masked_images and masks_jpg folder
    python examples/detectron2_SAM.py \
        --image_path $tmp_folder 



    
    # read -n1 -r -p "Done moving all images and creating masks . Press any key to continue..." key
    echo "Running $frame_name"
    if [ $i -eq 1 ]; then
    # For the first frame we use the sparse folder from COLMAP
        
        python examples/simple_trainer.py \
            --data_dir $tmp_folder \
            --data_factor 1 \
            --result_dir $output_folder \
            --test_every 8 \
            --max_steps 3000 \
            --save_steps 1 9000 21000 \
            --refine_start_iter 100 \
            --refine_stop_iter 15000 \
            --reset_every 500 \
            --refine_every 100 \
            --masked 
        # run="run0"
    fi
    if [ $i -ge 1 ]; then
        
        # Path to the JSON file from previous run
        last_run=$(printf "run%d" $((i - 2))) # run0 run1 run2 etc
        last_frame_name=$(printf "frame_%06d" $((1))) # frame_000001 ONLY ONE FOLDER subsequent frames in run1 run2 etc
        json_file=$(find $output_folder/$last_run/config_files -name "*.json" -type f -print -quit)
        next_run=$output_folder/$last_run/next_run # WE will use output_folder as next_run
        
        # Create the directory if it doesn't exist
        mkdir -p "$next_run"
        run_value=$((i - 1)) # 1 2 3 etc
        jq --argjson run_value "$run_value" 'del(.train_images, .test_images) | .run = $run_value | .refine_stop_iter = -1' "$json_file" > "$next_run/$(basename "$json_file")"

        # jq --argjson train_cam_ids "$train_cam_ids" 'del(.train_images, .test_images) | .run = 1 | .refine_stop_iter = -1 ' "$json_file" > "$next_run/$(basename "$json_file")"
        echo "Using JSON file: $next_run/$(basename "$json_file")"
        # Path to ckpt file from previous run
        ckpt_file=$(find $output_folder/$last_run/ckpts -name "ckpt_*.pt" -type f | sort -r -V | head -n 1)
        echo "Using ckpt file: $ckpt_file"
        # read -p "Press any key to continue..."
        python examples/simple_trainer.py \
            --ckpt $ckpt_file \
            --resume \

        fi
    # break
done