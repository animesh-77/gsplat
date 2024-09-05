import read_write_model as colmap
import argparse, os, glob, time

# 2 command line arguments path to images, path to sparse folder
parser = argparse.ArgumentParser()
parser.add_argument("--images_folder",default="", type=str, help="Path to the folder containing images")
parser.add_argument("--sparse_folder",default="", type=str, help="Path to the sparse folder")
parser.add_argument("--image_ext",default="PNG", type=str, help="Extension of the images")
args = parser.parse_args()
EXT= args.image_ext
# check if both are valid
if not os.path.exists(args.images_folder):
    print(f"Path to images folder: {args.images_folder} does not exist")
    exit(1)
if not os.path.exists(args.sparse_folder):
    print(f"Path to sparse folder: {args.sparse_folder} does not exist")
    exit(1)

# get a list of all iamges in images folder
all_images = glob.glob(os.path.join(args.images_folder, f'*.{EXT}'))
all_images= [os.path.basename(image) for image in all_images]
print(f"Total images in images folder: {len(all_images)}")



# read images.bin file and get all images
images_bin_file = os.path.join(args.sparse_folder, 'images.bin')
images_dict = colmap.read_images_binary(images_bin_file)

sparse_images= [image.name for image in images_dict.values()]
print(f"Total images in sparse folder: {len(sparse_images)}")

# find missing images and sort them
missing_images= list(set(all_images) - set(sparse_images))
missing_images.sort()
print(f"Total missing images: {len(missing_images)}")
for i, image in enumerate(missing_images):
    print(f"{i+1}. {image}")