import json, glob, os
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from segment_anything import sam_model_registry, SamPredictor


# add command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, default=False, 
                    help="Path to the images/ folder")

args = parser.parse_args()






cfg = get_cfg()
# Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
# Find a model from detectron2's model zoo.
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
detectron2_predictor = DefaultPredictor(cfg)
mask_categories = ["person", "handbag", "backpack", "suitcase", ]
with open("category2id.json", "r") as f:
    category2id = json.load(f)
mask_ids = [category2id[c] for c in mask_categories]



sam_checkpoint = "/cs/student/projects4/ml/2023/asrivast/SAM_models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

sam_predictor = SamPredictor(sam)


all_jpgs= glob.glob(f'{args.image_path}/*.jpg')
if len(all_jpgs) == 0:
    all_jpgs= glob.glob(f'{args.image_path}/*.JPG')
if len(all_jpgs) == 0:
    raise AssertionError("No images found to generate masks. Either do not use mask or FIX THIS")
print(f"total images: {len(all_jpgs)}.Generating masks for all images")

one_or_many= False # True for single point, False for multiple points
just_one= False # True to show only one image, False to not show any image and simply save the masks
for jpg in tqdm(all_jpgs):
    
    img = cv2.imread(jpg)
    outputs = detectron2_predictor(img)
    
    # Initialize empty bounding box list 
    # will append [x1, y1, x2, y2] for each mask
    bounding_boxes = []

    for i in range(len(outputs["instances"])):
        if outputs["instances"][i].pred_classes.cpu().numpy() in mask_ids:

            # mask = outputs["instances"][i].pred_masks.cpu().numpy()[0] # (H, W)
            bbox = outputs["instances"][i].pred_boxes.tensor.cpu().numpy()[0]  # (x1, y1, x2, y2)

            # Update bounding box mask if empty
            bounding_boxes.append(bbox)
    
    # if one_or_many is True:
    #     # Sample a single point from output_mask
    #     point_input = np.argwhere(output_mask)
    #     point_input = point_input[np.random.choice(len(point_input))]  # Randomly sample a point from the mask
    #     point_input = np.expand_dims(point_input, 0)  # Assuming SAM takes a batch of points of shape (1, 2) for a single point
    #     point_input = point_input[:, [1, 0]]  # Convert (y, x) to (x, y)
    #     point_label= np.array([1])
    #     # Run SAM on a single point
    #     sam_predictor.set_image(img)
    #     masks, scores, logits = sam_predictor.predict(
    #     point_coords=point_input,
    #     point_labels=point_label,
    #     multimask_output=True,) # we get 3 masks, 3 scores, and 3 logits

    #     final_mask = masks[0]
    #     for i, (mask, score) in enumerate(zip(masks, scores)):

    #         final_mask= np.logical_or(final_mask, mask)

    # Run SAM with bounding boxes
    sam_predictor.set_image(img)
    final_mask = np.zeros(img.shape[:2], dtype=bool)
    for bbox in bounding_boxes:

        masks, scores, logits = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box= bbox,
        multimask_output=True,) # we get 3 masks, 3 scores, and 3 logits

        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            # dilate mask before combining
            kernel = np.ones((5,5),np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        
            final_mask= np.logical_or(final_mask, mask)



    # Save mask
    if just_one is False: # save only when iterating over all images
        mask_name = jpg.replace("images", "masks_jpg")
        os.makedirs(os.path.dirname(mask_name), exist_ok=True)
        cv2.imwrite(mask_name, (final_mask * 255).astype(np.uint8))


        # save masked image
        masked_img= cv2.bitwise_and(img, img, mask= final_mask.astype(np.uint8))
        masked_img_name= jpg.replace("images", "masked_images_bbox")
        os.makedirs(os.path.dirname(masked_img_name), exist_ok=True)
        cv2.imwrite(masked_img_name, masked_img)
    