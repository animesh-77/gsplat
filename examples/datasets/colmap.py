import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
# from pycolmap import SceneManager
from scene_manager import SceneManager

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: Optional[int] = None,
        test_cam_ids: Optional[List[int]] = None,
        train_cam_ids: Optional[List[int]] = None, 
        masked: bool = False,
        config_files_path: str = "",
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.test_cam_ids = test_cam_ids
        self.train_cam_ids = train_cam_ids
        self.masked = masked
        
        # searches for a sparse/0 folder where the cameras.bin , images.bin and points3D.bin are stored
        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
            # when using COLMAP undistorter sparse/0 is not created only sparse
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        # SceneManager has attributes like cameras, images, points3D, points3D_ids etc
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()
        # manager.load_SMPL_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images # OrderedDict of Image objects
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective"
            ), f"Only support perspective camera model, got {type_}"

            params_dict[camera_id] = params

            # image size
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)

        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print(f"Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]
        # Remove the images/ prefix if it is there in all image names
        if all([n.startswith("images/") for n in image_names]):
            image_names = [n[len("images/") :] for n in image_names]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load images.
        if factor > 1:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        try:
            image_files.remove("Thumbs.db")
            colmap_files.remove("Thumbs.db")
        except ValueError:
            pass
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        # vertices from SMPL model
        # SMPL_points = manager.SMPL_points3D.astype(np.float64)

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds) # (4, 4)
            np.save(f"{config_files_path}/T1.npy", T1)
            camtoworlds = transform_cameras(T1, camtoworlds) # (N, 4, 4)
            points = transform_points(T1, points) # (M, 3)

            T2 = align_principle_axes(points) # (4, 4)
            np.save(f"{config_files_path}/T2.npy", T2)
            camtoworlds = transform_cameras(T2, camtoworlds) # (N, 4, 4)
            points = transform_points(T2, points) # (M, 3)

            transform = T2 @ T1 # (4, 4)
        else:
            transform = np.eye(4) # (4, 4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        if self.masked is True:
            # self.masks_paths = [img_path.replace("images", "masks_bb_jpg") for img_path in image_paths]
            self.masks_paths = [img_path.replace("images", "masks_jpg") for img_path in image_paths]
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)
        # self.SMPL_points = SMPL_points  # np.ndarray, (6890, 3)

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )
            self.Ks_dict[camera_id] = K_undist
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        # only the translation part of the extrinsic matrix
        scene_center = np.mean(camera_locations, axis=0)
        # centroid of the camera locations
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        # euclidean distance of each camera from the centroid
        self.scene_scale = np.max(dists)
        # maximum distance of a camera from the centroid


class Dataset:
    """A simple dataset class.
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    It must implement 3 functions __init__, __len__, and __getitem__
    """

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        masked: bool = False,
    ):
        """
        The __init__ function is run once when instantiating the Dataset object
        """
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        self.masked = masked
        # indices of images to use
        if split == "train":
            if self.parser.test_every is not None:
                # in default self.parser.test_every = 8, 
                # so for 9 images 8 will be training and 1 for testing
                self.indices = indices[indices % self.parser.test_every != 0]
                # if self.parser.test_every = 8
                # X 1 2 3 4 5 6 7 X 9 10 11 12 13 14 15 X
            else:
                self.indices = sorted(self.parser.train_cam_ids)
                
        else:
            # if self.parser.test_every = 8
            # 0 X X X X X X X 8 X X X X X X X 16
            if self.parser.test_every is not None:
                self.indices = indices[indices % self.parser.test_every == 0]
            else:
                self.indices = sorted(self.parser.test_cam_ids)

    def __len__(self):
        """
        The __len__ function returns the number of samples in our dataset
        or specifically the list of indices of images being used
        """
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        The __getitem__ function loads and returns a sample from the dataset at the given index idx
        here the returned sample is a dictionary
        --K:
        --camtoworld:
        --image:
        --image_id:
        and additonally if load_depths is True. In default case, it is False
        --points:
        --depths:
        """
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3] # (H, W, 3)
        # drops the alpha channel if it exists
        if self.masked is True:
            mask = imageio.imread(self.parser.masks_paths[index])  # Load the mask which is a binary image
            # (H, W) 0=background, 255=foreground
        else:
            mask = np.ones_like(image, dtype=np.uint8)*255
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            raise NotImplementedError
            mask = cv2.remap(mask, mapx, mapy, cv2.INTER_LINEAR)  # Undistort the mask
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]
            mask = mask[y : y + h, x : x + w]  # Crop the mask
            raise NotImplementedError

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            raise NotImplementedError
            mask = mask[y : y + self.patch_size, x : x + self.patch_size]  # Crop the mask
            K[0, 2] -= x
            K[1, 2] -= y

        # Apply the mask to the image
        idx = mask == 0 # all the background pixels
        masked_image= image.copy()
        masked_image[idx] = 0
    
        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(masked_image).float(),
            "image_id": item,  # the index of the image in the dataset
        }

        if self.load_depths:
            # in default case, self.load_depths is False
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            if self.patch_size is not None:
                points[:, 0] -= x
                points[:, 1] -= y
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
