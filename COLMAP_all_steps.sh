#!/bin/bash


database_path="/cs/student/projects4/ml/2023/asrivast/datasets/dress3_15/upright/sparse/database.db"
image_path="/cs/student/projects4/ml/2023/asrivast/datasets/dress3_15/upright/images"
mask_path="/cs/student/projects4/ml/2023/asrivast/datasets/dress3_15/upright/masks_jpg"
output_path="/cs/student/projects4/ml/2023/asrivast/datasets/dress3_15/upright/sparse"
max_image_size=6400
max_num_features=10000
project_path=$(dirname $database_path)

# if folder of database file does not exists then create it
# mkdir -p $(dirname $database_path)

# # Check if the database file exists and delete it
# if [ -f "$database_path" ]; then
#     rm "$database_path"
#     echo "Deleted database file: $database_path"
# else
#     echo "Database file not found: $database_path"
# fi


# colmap feature_extractor \
#     --database_path $database_path \
#     --image_path $image_path \
#     --SiftExtraction.use_gpu 1 \
#     --ImageReader.single_camera 0 \
#     --SiftExtraction.max_image_size $max_image_size \
#     --ImageReader.camera_model OPENCV \
#     --ImageReader.mask_path $mask_path \
#     # --ImageReader.single_camera_per_image 0 \


# echo "Done with feature extraction. Database file saved at: $database_path"
# read -n1 -r -p "Press any key to continue..." key
# # make output path if it doesn't exist
# mkdir -p $output_path

# colmap exhaustive_matcher \
#     --database_path $database_path \
#     --SiftMatching.use_gpu 1 \
#     --TwoViewGeometry.min_num_inliers 20 

# echo "Done with matching"
# read -n1 -r -p "Press any key to continue..." key

# colmap mapper \
#     --database_path $database_path \
#     --image_path $image_path \
#     --output_path $output_path \

echo "Mapping done. Output saved at: $output_path"
read -n1 -r -p "Press any key to continue..." key


colmap image_undistorter \
    --image_path $mask_path \
    --input_path $output_path/0 \
    --output_path $output_path/undistorted2 \
    --output_type COLMAP \
    # --max_image_size 4160 6240
    



read -n1 -r -p "Press any key to end..." key