#!/bin/bash
{
# Exit on error
set -e


dataset_path="/cs/student/projects4/ml/2023/asrivast/datasets/dress2_12"
# database_path="/cs/student/projects4/ml/2023/asrivast/datasets/portsmouth_video/frames/sparse_000050/database.db"
# image_path="/cs/student/projects4/ml/2023/asrivast/datasets/portsmouth_video/frames/000001"
# mask_path="/cs/student/projects4/ml/2023/asrivast/datasets/dress2_3/upright/masks_jpg"
# output_path="/cs/student/projects4/ml/2023/asrivast/datasets/portsmouth_video/frames/sparse_000050"
max_image_size=6400
max_num_features=10000

database_path="$dataset_path/sparse/database.db"
output_path="$dataset_path/sparse"
image_path="$dataset_path/images"
# if folder of database file does not exists then create it
mkdir -p $(dirname $database_path)

# merge models
mkdir -p $output_path/2
mkdir -p $output_path/3

colmap model_merger \
    --input_path1 $output_path/0 \
    --input_path2 $output_path/1 \
    --output_path $output_path/2
colmap bundle_adjuster \
    --input_path $output_path/2 \
    --output_path $output_path/3

# echo "Downsampling images by 2, 4 and 8 times."
# mkdir -p $dataset_path/dense/images_2
# magick mogrify $image -resize 50% -path $dataset_path/dense/images_2 $dataset_path/dense/images/*.JPG
# echo "Done with downsampling X2 ."

# mkdir -p $dataset_path/dense/images_4
# magick mogrify $image -resize 25% -path $dataset_path/dense/images_4 $dataset_path/dense/images/*.JPG
# echo "Done with downsampling X4 ."

# mkdir -p $dataset_path/dense/images_8
# magick mogrify $image -resize 12.5% -path $dataset_path/dense/images_8 $dataset_path/dense/images/*.JPG
# echo "Done with downsampling X8 ."

}
