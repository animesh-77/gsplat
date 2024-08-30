import glob, os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from segment_anything import sam_model_registry, SamPredictor

print("RUNNING GROUNDING SAM")
def segment(sam_predictor: SamPredictor, image: np.ndarray, detections) -> np.ndarray:
    sam_predictor.set_image(image)
    final_mask = np.zeros(image.shape[:2], dtype=bool)
    for detection in detections:
        box = detection[0]
        x, y, w, h = box
        bbox = np.array([x, y, x+w, y+h])
        masks, _, _ = sam_predictor.predict(point_coords=None,
                                            point_labels=None,
                                            box= bbox,
                                            multimask_output=True,)
        final_mask = np.logical_or(final_mask, masks[0])
        final_mask= np.logical_or(final_mask, masks[1])
        final_mask= np.logical_or(final_mask, masks[2])

    # logical_or of all masks
    return final_mask

def segment_bb(sam_predictor: SamPredictor, image: np.ndarray, detections) -> np.ndarray:
    sam_predictor.set_image(image)
    final_mask = np.zeros(image.shape[:2], dtype=bool)
    all_boxes = []
    for detection in detections:
        box = detection[0]
        x, y, w, h = box
        all_boxes.append([x, y, x + w, y + h])
    
    # Compute the hull of all bounding boxes
    if len(all_boxes) > 0:
        
        all_boxes = np.array(all_boxes)
        min_x = np.min(all_boxes[:, 0]).astype(int)
        min_y = np.min(all_boxes[:, 1]).astype(int)
        max_x = np.max(all_boxes[:, 2]).astype(int)
        max_y = np.max(all_boxes[:, 3]).astype(int)
    else:
        # mask everything
        min_x, min_y, max_x, max_y = 0, 0, image.shape[1], image.shape[0]
    
    # Draw the hull bounding box on the mask
    final_mask[min_y:max_y, min_x:max_x] = True

    # logical_or of all masks
    return final_mask

# add command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, default=".", 
                    help="Path to the fodler with images, masks_jpg and masked_images folders")
parser.add_argument("--classes", type= list, default=["person", "hands", ],
                                                    #   "handbag","purse", ],
                                                    #   "glasses", "white clothes", 
                                                    #   "sun glasses", "leather bag"],
                    help="List of classes to be masked")
parser.add_argument("--png", type=bool, default=False, help="If the images are in png format")
args = parser.parse_args()


# Grounding DINO imports
from groundingdino.util.inference import Model
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("/cs/student/projects4/ml/2023/asrivast/SAM_models/groundingdino_swint_ogc.pth")
# print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))
GROUNDING_DINO_CONFIG_PATH = os.path.join("/cs/student/projects4/ml/2023/asrivast/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
# print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)


# SAM imports
sam_checkpoint = "/cs/student/projects4/ml/2023/asrivast/SAM_models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)


CLASSES = args.classes
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.35

all_imgs= []
if args.png:
    all_pngs= glob.glob(f'{args.image_path}/*.png')
    if len(all_pngs) == 0:
        all_pngs= glob.glob(f'{args.image_path}/*.png')
    if len(all_pngs) == 0:
        raise AssertionError("No .png images found to generate masks. Either do not use mask or FIX THIS")
    print(f"total .png images: {len(all_pngs)}.Generating masks for all images")
    all_imgs= all_pngs

if not args.png:
    all_jpgs= glob.glob(f'{args.image_path}/*.jpg')
    if len(all_jpgs) == 0:
        all_jpgs= glob.glob(f'{args.image_path}/*.JPG')
    if len(all_jpgs) == 0:
        raise AssertionError("No .jpg images found to generate masks. Either do not use mask or FIX THIS")
    print(f"total .jpg images: {len(all_jpgs)}.Generating masks for all images")
    all_imgs= all_jpgs


# for jpg in tqdm(all_jpgs):
for jpg in tqdm(all_imgs):
    
    img = cv2.imread(jpg)


    # detect objects with DINO
    detections = grounding_dino_model.predict_with_classes(
        image=img,
        classes=CLASSES,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )


    final_mask = segment_bb(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        detections=detections
    )

    # output_mask= np.zeros(img.shape[:2], dtype=bool)
    # for detection in detections:
    #     confidence, class_id, mask = detection[2], detection[3], detection.mask

    #     # Dilate the mask before adding to the output mask
    # kernel = np.ones((5,5),np.uint8)
    # final_mask= cv2.erode(final_mask.astype(np.uint8), kernel, iterations=2)
    

    mask_name = jpg.replace("images", "masks").replace(".JPG", ".png").replace(".jpg", ".png")
    os.makedirs(os.path.dirname(mask_name), exist_ok=True)
    cv2.imwrite(mask_name, (final_mask * 255).astype(np.uint8))


    # save masked image
    # masked_img= cv2.bitwise_and(img, img, mask= final_mask.astype(np.uint8))
    # masked_img_name= jpg.replace("images", "masked_images_dino")
    # os.makedirs(os.path.dirname(masked_img_name), exist_ok=True)
    # cv2.imwrite(masked_img_name, masked_img)
    # break

mask_name = os.path.dirname(jpg.replace("images", "masks"))
print(f"Mask saved at         {mask_name}")
masked_img_name= os.path.dirname(jpg.replace("images", "masked_images_dino"))
print(f"Masked image saved at {masked_img_name}")
print("DONE RUNNING GROUNDING SAM")
