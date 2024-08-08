from datetime import datetime
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import nerfview
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from datasets import normalize
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    knn,
    normalized_quat_to_rotmat,
    rgb_to_sh,
    set_random_seed,
)

from gsplat.rendering import rasterization
from plyfile import PlyData, PlyElement
import vedo
from pathlib import Path
from scipy.spatial import cKDTree


@dataclass
class Config:
    # dataclass removes the need to add the __init__ method
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # Resume training from previous checkpoint
    resume: bool = False
    # which run are we on
    run: int = 0
    # whether to use masks of the images
    masked: bool = False
    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    # for 1 it uses images/ for 2 it uses images_2/ , for 4 it uses images_4/
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    # So in 9 images :- 8 images for training and 1 image for testing
    test_every: Optional[int] = None
    # add feature to directly call test cameras by index
    test_cam_ids: Optional[List[int]] = field(default_factory=lambda: [2, 15, 30 ,45])
    # add feature to directly call train cameras by index
    train_cam_ids: Optional[List[int]] = field(default_factory=lambda: [1, 50])
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    # used as factor in adjust_steps()
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.005
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100
    # Max gaussians to keep in the scene
    max_gaussians: int = 30_000

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)

    def read_json_and_make_config(self) -> "Config":
        """
        Read the json file and make a config object
        which .json to read is specified by which ckeckpoint is being loaded as a .pt file
        """
        step= re.split("[_.]", self.ckpt)[-2]
        runn_dir= os.path.dirname(os.path.dirname(self.ckpt))
        config_file_json= os.path.join(runn_dir, "next_run", f"save_{step}.json")
        with open(config_file_json, 'r') as f:
            config = json.load(f)
        cfg= Config(**config)
        cfg.ckpt= self.ckpt
        cfg.resume= True
        return cfg

def load_SMPL_file_obj(data_dir: str, T1: str, T2: str):
    """
    Load the SMPL .obj mesh and return only the vertices
    the faces are not needed
    """
    T1= np.load(T1)
    T2= np.load(T2)
    
    data_dir= Path(data_dir)
    obj_file = list(data_dir.rglob("*.obj"))[0]
    mesh = vedo.Mesh(obj_file.as_posix())
    xyz= mesh.vertices

    xyz= normalize.transform_points(T1, xyz)
    xyz= normalize.transform_points(T2, xyz)
    return xyz

def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, torch.optim.Optimizer]:
    """
    splats is a dictionary of parameters
        - means3d
        - scales
        - quats
        - opacities
        - sh0
        - shN
        - features (not by deafult)
        - colors (not by default)
    
    optimizers is a list of optimizers for each parameter of splats
    """
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float() # (N, 3)
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float() # (N, 3)
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1) # [N, 3]
        rgbs = torch.rand((init_num_pts, 3)) # [N, 3]
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")
    # points and rgb are torch tensors
    N = points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    # using sklearn for knn
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means3d", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # by default feature_dim is None
        # color is SH coefficients.
        # by default sh_degree is 3
        # so k= 16
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, 16, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs) # shape of rgbs is [N, 3]
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # parameters are stored in a dictionary
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizers = [
        (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size), "name": name}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    ]
    # optimizers is a list of optimizers for each parameter
    # either all will be Adam or all will be SparseAdam
    return splats, optimizers

def create_optimizers_only(
    splats: torch.nn.ParameterDict,
    # params: List[Tuple[str, torch.nn.Parameter, float]],
    scene_scale: float = 1.0,
    batch_size: int = 1,
    sparse_grad: bool = False,
    device: str = "cuda",
) -> torch.optim.Optimizer:
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1

    lr_map= {
        "means3d": 1.6e-4* scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
        "features": 2.5e-3,
        "colors": 2.5e-3,
    }
    params= []
    for name, value in splats.items():
        params.append((name, value, lr_map[name]))


    optimizers = [
        (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name].to(device), "lr": lr * math.sqrt(batch_size), "name": name}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    ]
    return optimizers

class Runner:
    """Engine for training and testing.
    This is the class that has the attricutes
    - splats:
    - optimizers:
    """

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/run{cfg.run}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/run{cfg.run}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/run{cfg.run}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        self.ply_dir = f"{cfg.result_dir}/run{cfg.run}/ply_files"
        os.makedirs(self.ply_dir, exist_ok=True)
        self.config_dir = f"{cfg.result_dir}/run{cfg.run}/config_files"
        os.makedirs(self.config_dir, exist_ok=True)
        # Tensorboard
        now = datetime.now()
        current_time = f"{now.strftime('%b')}-{now.day}__{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
        log_dir = f"{cfg.result_dir}/run{cfg.run}/tb/{current_time}/"
        self.writer = SummaryWriter(log_dir= log_dir)

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True, # this changes the scale/orientation of the scene
            test_every=cfg.test_every,
            test_cam_ids=cfg.test_cam_ids,
            train_cam_ids=cfg.train_cam_ids,
            masked=cfg.masked,
            config_files_path= self.config_dir,
        )
        # Parser is a COLMAP parser that reads the images and the 3D points from the COLMAP model
        # It has attributes like:
        # - self.cameras: 
        # - self.images:
        # - self.points3D:
        # - self.points3D_ids:
        # - self.points3D_id_to_images:
        # :NOTE points are normalised (scaled/rotated) so they are NOT at the 
        # same position as COLMAP/sparse/0/points3D.bin
        self.trainset = Dataset(
            self.parser,
            split="train", # split is either "train" or "val" 
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            masked= cfg.masked,
        )
        self.valset = Dataset(self.parser, 
                              split="val",
                              masked= cfg.masked,)
        # here "val" so remaining images are used for testing
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        # by default cfg.app_opt is False so feature_dim is None
        # actual splats and optimizers 
        # different optimizers for each parameter
        # self.splats is a dictionary
        if cfg.resume is False:
            # we are starting from scratch run0
            self.splats, self.optimizers = create_splats_with_optimizers(
                self.parser,
                init_type=cfg.init_type,
                init_num_pts=cfg.init_num_pts,
                init_extent=cfg.init_extent,
                init_opacity=cfg.init_opa,
                init_scale=cfg.init_scale,
                scene_scale=self.scene_scale,
                sh_degree=cfg.sh_degree, # by default 3
                sparse_grad=cfg.sparse_grad,
                batch_size=cfg.batch_size,
                feature_dim=feature_dim,
                device=self.device,
            )
            try:
                self.SMPL= load_SMPL_file_obj(self.cfg.data_dir, f"{self.config_dir}/T1.npy", f"{self.config_dir}/T2.npy")
                self.SMPL= torch.from_numpy(self.SMPL).to(self.device)
            except IndexError:
                self.SMPL= None
                print("SMPL .obj file not found")
        else:
            # resume is True, we are resuming from a checkpoint
            # we are resuming from a checkpoint
            # load the checkpoint
            print("Resuming from checkpoint:", cfg.ckpt)
            ckpt = torch.load(cfg.ckpt, map_location="cuda")
            self.splats = torch.nn.ParameterDict(ckpt["splats"]).to(self.device)
            # create optimizers for the splats
            self.optimizers = create_optimizers_only(
                splats=self.splats,
                scene_scale=self.scene_scale,
                batch_size=cfg.batch_size,
                sparse_grad=cfg.sparse_grad,
                device=self.device,
            )
            try:
                self.SMPL= load_SMPL_file_obj(self.cfg.data_dir, f"{self.config_dir}/T1.npy", f"{self.config_dir}/T2.npy")
                self.SMPL= torch.from_numpy(self.SMPL).to(self.device)
            except IndexError:
                self.SMPL= None
                print("SMPL .obj file not found")

        print("Model initialized. Number of GS:", len(self.splats["means3d"]))

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

        # Running stats for prunning & growing.
        n_gauss = len(self.splats["means3d"])
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=self.device),  # norm of the gradient
            "count": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
        }

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means3d"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )
        return render_colors, render_alphas, info

    def splat_scale_loss(self) -> Tensor:
        """
        for every GS, calculate the scale loss which is the ratio of max and min scale of each GS

        """
        scales = torch.exp(self.splats["scales"])
        scale_loss = torch.log(scales.max(dim=0).values / scales.min(dim=0).values).mean()
        return scale_loss
    
    def near_splat_loss(self) -> Tensor:
        """
        For every guassian splat, get the nearest splat and calculate the difference in their colors and rotations
        """
        tree = cKDTree(self.splats["means3d"].cpu().detach().numpy())
        _, indices = tree.query(self.splats["means3d"].cpu().detach().numpy(), k=4)
        # distances (N, k=4)
        # indices (N, k=4)
        indices = torch.from_numpy(indices).to(self.device) # (N, k=4)
        # get the nearest splats
        nearest_splats = self.splats["means3d"][indices[:, 1:]] # (N, k=4-1, 3)
        nearest_quats = self.splats["quats"][indices[:, 1:]] # (N, k=4-1, 4)
        # nearest_colors = self.splats["colors"][indices[:, 1]]

        # Expand current splats and quats to match nearest splats and quats dimensions for broadcasting
        current_splats = self.splats["means3d"].unsqueeze(1).expand_as(nearest_splats)  # (N, k-1, 3)
        current_quats = self.splats["quats"].unsqueeze(1).expand_as(nearest_quats)      # (N, k-1, 4)

        # Calculate distances and quaternion differences
        distances = torch.norm(nearest_splats - current_splats, dim=2)  # (N, k-1)
        quat_diffs = torch.norm(nearest_quats - current_quats, dim=2)   # (N, k-1)

        # Calculate the loss using vectorized operations
        loss = (distances * quat_diffs).mean()
        return loss
    
    def SMPL_loss(self) -> Tensor:
        """
        for each splat get the closest point on the SMPL mesh and calculate the loss
        """
        tree = cKDTree(self.SMPL.cpu().detach().numpy())
        _, indices = tree.query(self.splats["means3d"].cpu().detach().numpy(), k=1)
        # distances (N, k=1)
        # indices (N, k=1)
        indices = torch.from_numpy(indices).to(self.device)
        nearest_points = self.SMPL[indices] # (N, 3)
        distances = torch.norm(nearest_points - self.splats["means3d"], dim=1) # (N,)
        loss = distances.mean()
        return loss

    @torch.no_grad()
    def total_splats(self) -> int:
        return len(self.splats["means3d"])
    
    @torch.no_grad()
    def SMPL_far_filter(self) :
        """
        remove gaussians which are far from the SMPL mesh
        """
        tree = cKDTree(self.SMPL.cpu().detach().numpy())
        _, indices = tree.query(self.splats["means3d"].cpu().detach().numpy(), k=1)
        # distances (N, k=1)
        # indices (N, k=1)
        indices = torch.from_numpy(indices).to(self.device)
        nearest_points = self.SMPL[indices]

    def train(self):
        """
        self is an instance of class Runner
        it ahs attributes like:
        - self.splats
          - means3d
          - scales
        """
        cfg = self.cfg
        device = self.device

        # Dump cfg
        # let this be. This json serves as a record of the config used for run0
        # save .json for index of all images. This can be used for next runs
        if self.cfg.run == 0:
            dict_to_save = self.cfg.__dict__
            dict_to_save["train_images"]= [(int(i), self.parser.image_names[i]) for i in self.trainset.indices]
            dict_to_save["test_images"]= [(int(i), self.parser.image_names[i]) for i in self.valset.indices]
            with open(f"{cfg.result_dir}/cfg.json", "w") as f:
                json.dump(dict_to_save, f, indent=4)
            self.save_img_list()


        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means3d has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers[0], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        # custom progress bar which will be updated manually
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                # depth loss is False by default
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            info["means2d"].retain_grad()  # used for running stats

            # loss
            # colors is the rendered image
            # pixels is the ground truth image
            l1loss = F.l1_loss(colors, pixels)
            l2loss = F.mse_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            # lpipsloss = self.lpips(pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)/255.0)
            # scale_loss = self.splat_scale_loss()
            # nearby_splat_loss = self.near_splat_loss()
            # SMPL_loss = self.SMPL_loss()
            loss = l1loss * (1.0 - cfg.ssim_lambda) + \
                ssimloss * cfg.ssim_lambda +  l2loss * 0.3  \
            #     scale_loss * 0.7 + \
            #    + \
            #     lpipsloss * 0.7 + \
            #     SMPL_loss * 0.7 
                # nearby_splat_loss
            # loss= SMPL_loss * 0.3+ nearby_splat_loss * 0.7
            if cfg.depth_loss:
                # cfg.depth_loss is False by default
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0 or step == max_steps - 1: 
                # tensorboard writer here
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/L1_loss", l1loss.item(), step)
                self.writer.add_scalar("train/SSIM_loss", ssimloss.item(), step)
                self.writer.add_scalar(
                    "train/num_GS", len(self.splats["means3d"]), step
                )

                with torch.no_grad():
                    max_splat_scale = torch.abs(torch.max(self.splats["scales"], dim= 1).values) # 
                    min_splat_scale = torch.abs(torch.min(self.splats["scales"], dim= 1).values) # min scale of all splats
                    # add histogram of ratio of max and min scale
                    ratio= max_splat_scale/ min_splat_scale
                    self.writer.add_histogram("train/splats_scale", ratio, step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            if self.total_splats() > cfg.max_gaussians and cfg.refine_stop_iter > step:
                print("Max splats reached. Stopping refinement.")
                cfg.refine_stop_iter = step
            
            # update running stats for prunning & growing
            if step < cfg.refine_stop_iter:
                self.update_running_stats(info)

                if step > cfg.refine_start_iter and step % cfg.refine_every == 0:
                    grads = self.running_stats["grad2d"] / self.running_stats[
                        "count"
                    ].clamp_min(1)

                    # grow GSs
                    is_grad_high = grads >= cfg.grow_grad2d
                    is_small = (
                        torch.exp(self.splats["scales"]).max(dim=-1).values
                        <= cfg.grow_scale3d * self.scene_scale
                    )
                    is_dupli = is_grad_high & is_small
                    n_dupli = is_dupli.sum().item()
                    self.refine_duplicate(is_dupli)

                    is_split = is_grad_high & ~is_small
                    is_split = torch.cat(
                        [
                            is_split,
                            # new GSs added by duplication will not be split
                            torch.zeros(n_dupli, device=device, dtype=torch.bool),
                        ]
                    )
                    n_split = is_split.sum().item()
                    self.refine_split(is_split)
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # prune GSs
                    is_prune = torch.sigmoid(self.splats["opacities"]) < cfg.prune_opa
                    if step > cfg.reset_every:
                        # The official code also implements sreen-size pruning but
                        # it's actually not being used due to a bug:
                        # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
                        is_too_big = (
                            torch.exp(self.splats["scales"]).max(dim=-1).values
                            > cfg.prune_scale3d * self.scene_scale
                        )
                        is_prune = is_prune | is_too_big
                    n_prune = is_prune.sum().item()
                    self.refine_keep(~is_prune)
                    print(
                        f"Step {step}: {n_prune} GSs pruned. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # reset running stats
                    self.running_stats["grad2d"].zero_()
                    self.running_stats["count"].zero_()

                if step % cfg.reset_every == 0:
                    self.reset_opa(cfg.prune_opa * 2.0)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # at save steps we save the model, config files
            # we can later continue training from any of these checkpoints
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means3d"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/save_step_{step}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

                # save the ply file only on save steps not on eval steps
                # on eval steps we only want to evaluate the model

                self.save_ply(step)
                print(f"PLY file saved to {self.ply_dir}/save_{step}.ply")
                # save the config file
            if step == max_steps - 1: # last step
                self.save_config(step)

            # evaluation done on the evaluation cameras
            # for all eval steps and last step
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                # run eavlulation on eval_steps and just before last step
                self.eval(step)
                # self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def update_running_stats(self, info: Dict):
        """Update running stats."""
        cfg = self.cfg

        # normalize grads to [-1, 1] screen space
        if cfg.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * cfg.batch_size
        grads[..., 1] *= info["height"] / 2.0 * cfg.batch_size
        if cfg.packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz] or None
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )

    @torch.no_grad()
    def reset_opa(self, value: float = 0.01):
        """Utility function to reset opacities."""
        opacities = torch.clamp(
            self.splats["opacities"], max=torch.logit(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "opacities":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(opacities)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_split(self, mask: Tensor):
        """Utility function to grow GSs."""
        device = self.device

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = torch.exp(self.splats["scales"][sel])  # [N, 3]
        quats = F.normalize(self.splats["quats"][sel], dim=-1)  # [N, 4]
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            0.5* torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]

        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
                else:
                    repeats = [2] + [1] * (p.dim() - 1)
                    p_split = p[sel].repeat(repeats)
                p_new = torch.cat([p[rest], p_split])
                p_new = torch.nn.Parameter(p_new)
                # update optimizer
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key == "step":
                        continue
                    v = p_state[key]
                    # new params are assigned with zero optimizer states
                    # (worth investigating it)
                    v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
                    p_state[key] = torch.cat([v[rest], v_split])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if v is None:
                continue
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            self.running_stats[k] = torch.cat((v[rest], v_new))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_duplicate(self, mask: Tensor):
        """Unility function to duplicate GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sel), *v.shape[1:]), device=self.device
                        )
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_keep(self, mask: Tensor):
        """Unility function to prune GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = p_state[key][sel]
                p_new = torch.nn.Parameter(p[sel])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[sel]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def eval(self, step: int):
        """
        Entry for evaluation.
        Render image for each evaluation cameras
        and calculate PSNR, SSIM, LPIPS
        """
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            # original image
            height, width = pixels.shape[1:3]
            image_id= self.valset.indices[i]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            # colors is the rendered image
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            # left side is the original image and right side is the rendered image
            imageio.imwrite(
                f"{self.render_dir}/val_{step}_cam_{image_id}.png", (canvas * 255).astype(np.uint8)
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        # report the mean value over all validation images
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means3d'])}"
        )
        # save stats as json
        stats = {
            "PSNR": psnr.item(),
            "SSIM": ssim.item(),
            "LPIPS": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means3d"]),
        }
        with open(f"{self.stats_dir}/eval_step_{step}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()


    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/run{cfg.run}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()

    # Experimental
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.splats["sh0"].shape[1]*self.splats["sh0"].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.splats["shN"].shape[1]*self.splats["shN"].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.splats["scales"].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.splats["quats"].shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # Experimental
    @torch.no_grad()
    def save_ply(self, steps: int):
        """
        Model saved as .ply file on save steps
        self.ply_dir is the path to the folder we want to save the ply files
        """
        os.makedirs(os.path.dirname(self.ply_dir), exist_ok=True)

        xyz = self.splats["means3d"].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.splats["opacities"].detach().unsqueeze(-1).cpu().numpy()
        scale = self.splats["scales"].detach().cpu().numpy()
        rotation = self.splats["quats"].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        ply_file_path = os.path.join(self.ply_dir, f"step_{steps}.ply")
        PlyData([el]).write(ply_file_path)

    def save_config(self, step: int):
        """
        save the config files for this run so they can be used for the next run
        only done at last iteration
        an additonal key is added to the config file with the list of test images and train images
        this key is deleted when resuming training
        """
        config_file_path = os.path.join(self.config_dir, f"save_{step}.json")
        os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
        dict_to_save= self.cfg.__dict__
        # in this dict add a key with list of image names
        dict_to_save["train_images"]= [(int(i), self.parser.image_names[i]) for i in self.trainset.indices]
        dict_to_save["test_images"]= [(int(i), self.parser.image_names[i]) for i in self.valset.indices]
        with open(config_file_path, 'w') as f:
            json.dump(dict_to_save, f, indent=4)
    def save_img_list(self):
        """
        save the list of all images and their index only once
        """
        img_index_dict= {key: value for key, value in enumerate(self.parser.image_names)}
        with open(f"{self.cfg.result_dir}/img_index.json", "w") as f:
            json.dump(img_index_dict, f, indent=4)
        
def main(cfg: Config):
    # cfg is an instance of class Config
    

    if (cfg.ckpt is not None) and (cfg.resume is True):
        # resume training from the from some checkpoint
        # a new run_id folder is creataed where all the results will be saved
        cfg= cfg.read_json_and_make_config()
        # cfg.run += 1 # no need to increase since it is done in bash script
        ckpt = torch.load(cfg.ckpt, map_location= "cuda")
        runner = Runner(cfg)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        
        runner.train()

    else:
        runner = Runner(cfg)
        if (cfg.ckpt is not None) and (cfg.resume_training is False):
            # run evaluation only and no training done
            # no .ply or config files .json files are saved
            pass
            runner.render_traj(step=ckpt["step"])
            # save the ply file
            iterations = re.split("[._]", os.path.basename(cfg.ckpt))[1]
            ply_file_path= os.path.join(cfg.result_dir, "ply_files", f"{iterations}.ply")
            runner.save_ply(ply_file_path)
            print(f"PLY file saved to {ply_file_path}")
            # save the config file
            config_file_path= os.path.join(cfg.result_dir, "config_files", f"{iterations}.json")
            runner.save_config(config_file_path)

        else:
            # run full training loop
            runner.train()

    if not cfg.disable_viewer:
        try:
            print("Viewer running... Ctrl+C to exit.")
            time.sleep(10)
        except KeyboardInterrupt:
            print("Viewer stopped.")

if __name__ == "__main__":
    # tyro.cli removes the need to manually add argparse 
    cfg = tyro.cli(Config)
    # cfg is an instance of class Config with 1 function
    # scales some of the arguments
    cfg.adjust_steps(factor= cfg.steps_scaler)
    main(cfg)
