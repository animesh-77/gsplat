#!/bin/bash
{
# Exit on error
set -e


dataset_path="/cs/student/projects4/ml/2023/asrivast/datasets/dress1_2"
# database_path="/cs/student/projects4/ml/2023/asrivast/datasets/portsmouth_video/frames/sparse_000050/database.db"
# image_path="/cs/student/projects4/ml/2023/asrivast/datasets/portsmouth_video/frames/000001"
# mask_path="/cs/student/projects4/ml/2023/asrivast/datasets/dress2_3/upright/masks_jpg"
# output_path="/cs/student/projects4/ml/2023/asrivast/datasets/portsmouth_video/frames/sparse_000050"
max_image_size=6400
max_num_features=10000

database_path="$dataset_path/sparse_png/database.db"
output_path="$dataset_path/sparse_png"
image_path="$dataset_path/images_png"
# if folder of database file does not exists then create it
mkdir -p $(dirname $database_path)

# Check if the database file exists and delete it
if [ -f "$database_path" ]; then
# Ask user if they want to delete the database file and start new or skip feature extraction
    read -p "Database file already exists. Do you want to delete it and start fresh? y/n: " choice
    if [[ $choice == "y" ]]; then
        rm "$database_path"
        echo "Deleted database file: $database_path"
    fi
else
    echo "Database file not found. Will start fresh."
fi


if [[$choice == "y"]]; then
    read -n1 -r -p "Press any key to continue with feature extraction..." key

    colmap feature_extractor \
        --database_path $database_path \
        --image_path $image_path \
        --SiftExtraction.use_gpu 1 \
        --ImageReader.single_camera 0 \
        --SiftExtraction.max_image_size $max_image_size \
        --SiftExtraction.max_num_features $max_num_features \
        --ImageReader.camera_model OPENCV 
        # --SiftExtraction.estimate_affine_shape 1 \
        # --SiftExtraction.domain_size_pooling 0 
        # --ImageReader.mask_path $mask_path \
        # --ImageReader.single_camera_per_image 0 \
else
    echo "Skipped feature extraction."
fi


echo "Done with feature extraction. Database file saved at: $database_path"
read -n1 -r -p "Press any key to continue with exhaustive matcher..." key
# # make output path if it doesn't exist
mkdir -p $output_path

colmap exhaustive_matcher \
    --database_path $database_path \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.max_num_matches 36000 \
    --TwoViewGeometry.min_num_inliers 50 \
    --SiftMatching.guided_matching 1

echo "Done with exhaustive matcher"
read -n1 -r -p "Press any key to continue with mapper..." key

colmap mapper \
    --database_path $database_path \
    --image_path $image_path \
    --output_path $output_path \
    --Mapper.multiple_models 1 \
    --Mapper.ba_global_function_tolerance 0.000001 \
    --Mapper.init_image_id1 1
    # --Mapper.min_num_matches 15 \
    # --Mapper.init_min_num_inliers 40

# get all folders inside $dataset_path/sparse and convert all the .bin files to .txt

for folder in "$output_path"/*; 
do
    if [ -d "$folder" ]; then
        colmap model_converter \
            --input_path "$folder" \
            --output_path "$folder" \
            --output_type TXT
    fi
done

echo "Done with mapper."
echo "CHECK THERE IS ONLY ONE FOLDER 0 inside $output_path. "
echo "IF NOT CHOOSE THE BEST ONE MANUALLY."
read -n1 -r -p "Press any key to continue to see missing images. Ctrl+c to terminate" key

python COLMAP_missing_images.py --images_folder $image_path \
    --sparse_folder $output_path/0 \
    --image_ext PNG

read -n1 -r -p "Press any key to continue with image undistorter. Ctrl+c to terminate" key

# :NOTE: Only one reonstructed model is used in this step.
colmap image_undistorter \
    --image_path $image_path \
    --input_path $output_path/0 \
    --output_path $dataset_path/dense_png \
    --output_type COLMAP \
    # --max_image_size 4160 6240
    
echo "Done with image undistortion."
read -n1 -r -p "Press any key to continue with model converter" key

colmap model_converter \
        --input_path $dataset_path/dense_png/sparse \
        --output_path $dataset_path/dense_png/sparse \
        --output_type TXT

echo "Done with model conversion to txt."
read -p "Continue with next steps, patch_match_stereo and stereo_fusion ? y/n: " choice

if [[ $choice == "y" ]]; then
    

    colmap patch_match_stereo \
        --workspace_path $dataset_path/dense_png \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true

    colmap stereo_fusion \
        --workspace_path $dataset_path/dense_png \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path $dataset_path/dense_png/fused.ply
fi


# make a folder output_path/dense_png/images_2 to store downsampled images
echo "Downsampling images by 2, 4 and 8 times."
mkdir -p $dataset_path/dense_png/images_2
magick mogrify $image -resize 50% -path $dataset_path/dense_png/images_2 $dataset_path/dense_png/images/*.PNG
echo "Done with downsampling X2 ."

mkdir -p $dataset_path/dense_png/images_4
magick mogrify $image -resize 25% -path $dataset_path/dense_png/images_4 $dataset_path/dense_png/images/*.PNG
echo "Done with downsampling X4 ."

mkdir -p $dataset_path/dense_png/images_8
magick mogrify $image -resize 12.5% -path $dataset_path/dense_png/images_8 $dataset_path/dense_png/images/*.PNG
echo "Done with downsampling X8 ."

exit
}