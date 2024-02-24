import logging
import os
import json

# import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import ToPILImage, ToTensor, Resize, Compose, Normalize
import lightning.pytorch as pl


from PIL import Image
import open3d as o3d
import cv2
from torch.utils.data import DataLoader


class BOP(Dataset):
    """
    BOP dataset class for loading data from the BOP dataset.

    Args:
        root_dir (str): Root directory of the dataset.
        phase (str): Phase of the dataset, either 'train', 'val', or 'test'.
        transform (callable, optional): Optional transform to be applied on a sample.
        obj_id_list (list, optional): List of object IDs to include in the dataset. If empty, includes all object IDs from 1 to 30.
        vis_ratio (float, optional): Minimum visibility ratio for objects to be included in the dataset.
        remove_repeated_objects (bool, optional): Whether to remove scenes with repeated objects.
    """

    def __init__(self, root_dir, phase, transform=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.root = None
        if isinstance(root_dir, str):
            if os.path.exists(f"{root_dir}"):
                self.root = root_dir

        if self.root is None:
            raise AssertionError(f"Dataset not found in {root_dir}")
        else:
            self.logger.info(f"Loading data from {self.root}")

        self.depth_threshold = 3000
        # set the directories related to the T-Less dataset
        self.train_dir = os.path.join(self.root, "train_pbr")
        self.test_dir = os.path.join(
            self.root, "tless_test_primesense_bop19", "test_primesense"
        )
        self.cad_dir = os.path.join(self.root, "tless_models", "models_cad")

        # each directory is related to a specific camera and scene setup
        self.selected_scenes = []
        assert phase in ["train", "val", "test"]
        base_dir = self.train_dir if phase in ["train"] else self.test_dir

        self.create_data_list(base_dir)

        if transform is None:
            self.transform = Compose(
                [
                    Resize((448, 448)),
                    ToTensor(),
                ]
            )
        else:
            self.transform = transform

        self.phase = phase

    def __len__(self):
        """
        Returns the number of scenes in the dataset.

        Returns:
            int: Number of scenes in the dataset.
        """
        return len(self.selected_scenes)

    def __getitem__(self, ind):
        """
        Returns a sample from the dataset at the given index.

        Args:
            ind (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the RGB image, XYZ map, RGB image path, and depth image path.
        """
        rgb_ext = ".jpg" if self.phase == "train" else ".png"
        rgb_path = os.path.join(
            self.selected_scenes[ind]["dir"],
            "rgb",
            str(self.selected_scenes[ind]["scene_id"]).zfill(6) + rgb_ext,
        )
        depth_path = os.path.join(
            self.selected_scenes[ind]["dir"],
            "depth",
            str(self.selected_scenes[ind]["scene_id"]).zfill(6) + ".png",
        )

        rgb = Image.open(rgb_path)
        rgb = self.transform(rgb)
        rgb = BOP.norm_transform(rgb)

        cam_K = np.array(self.selected_scenes[ind]["cam_K"]).reshape(3, 3)
        cam_depth_scale = 1 / self.selected_scenes[ind]["depth_scale"]
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_image = torch.from_numpy(depth_image.astype(float))
        xyz_map = self.calculate_xyz_map(depth_image, cam_K, cam_depth_scale)
        xyz_map = ToPILImage()(xyz_map)
        xyz_map = self.transform(xyz_map)

        return {
            "rgb": rgb,
            "xyz_map": xyz_map,
            "rgb_xyz": torch.cat([rgb, xyz_map], dim=2),
            "rgb_path": rgb_path,
            "depth_path": depth_path,
        }

    def calculate_xyz_map(self, depth_image, camera_K, cam_depth_scale):
        """
        Calculates the XYZ map from the depth image.

        Args:
            depth_image (torch.Tensor): Depth image.
            camera_K (np.ndarray): Camera intrinsic matrix.
            cam_depth_scale (float): Depth scale factor.

        Returns:
            torch.Tensor: XYZ map.
        """
        # Define the intrinsic parameters
        fx = camera_K[0, 0]  # Focal length x
        fy = camera_K[1, 1]  # Focal length y
        cx = camera_K[0, 2]  # Center x
        cy = camera_K[1, 2]  # Center y

        # Create a meshgrid for pixel coordinates
        height, width = depth_image.shape

        x = torch.linspace(0, height - 1, height)
        y = torch.linspace(0, width - 1, width)
        x, y = torch.meshgrid(x, y, indexing="ij")

        # Convert depth image to meters
        # if the depth image is in millimeters
        depth_image_meters = depth_image / cam_depth_scale

        # Calculate the XYZ coordinates
        X = (x - cx) * depth_image_meters / fx
        Y = (y - cy) * depth_image_meters / fy
        Z = depth_image_meters
        # Normalize the values
        Z[Z > self.depth_threshold] = self.depth_threshold
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())
        Z = (Z - Z.min()) / (self.depth_threshold - Z.min())
        # Stack X, Y, Z to create the 3D map
        xyz_map = torch.dstack((X, Y, Z))

        return xyz_map.permute((2, 0, 1)).float()

    def create_data_list(self, base_dir):
        """
        Creates a list of selected scenes based on the given parameters.

        Args:
            obj_id_list (list): List of object IDs to include in the dataset.
            vis_ratio (float): Minimum visibility ratio for objects to be included in the dataset.
            remove_repeated_objects (bool): Whether to remove scenes with repeated objects.
            base_dir (str): Base directory of the dataset.
        """
        setup_dirs = [
            name
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir))
        ]
        for setup_dir in setup_dirs:
            scene_gt_path = os.path.join(base_dir, setup_dir, "scene_gt.json")
            camera_gt_path = os.path.join(base_dir, setup_dir, "scene_camera.json")
            scene_info_path = os.path.join(base_dir, setup_dir, "scene_gt_info.json")
            with open(scene_gt_path) as scenes_f, open(
                camera_gt_path
            ) as camera_f, open(scene_info_path) as scene_info_f:
                scenes_dic = json.load(scenes_f)
                cameras_dic = json.load(camera_f)
                for scene_id, _ in scenes_dic.items():
                    data_dic = {
                        "dir": os.path.join(base_dir, setup_dir),
                        "scene_id": int(scene_id),
                    }
                    data_dic.update(cameras_dic[scene_id])
                    self.selected_scenes.append(data_dic)

    invNorm_transform = Compose(
        [
            Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )

    norm_transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class BOP_feature(BOP):
    """
    A class representing a BOP feature dataset.

    Args:
        root_dir (str): The root directory of the dataset.
        phase (str): The phase of the dataset (e.g., train, val, test).

    Attributes:
        root_dir (str): The root directory of the dataset.
        phase (str): The phase of the dataset (e.g., train, val, test).
        selected_scenes (list): A list of selected scenes.

    Methods:
        __getitem__(ind): Retrieves the RGB and depth features for a given index.

    """

    def __init__(self, root_dir, phase):
        super().__init__(root_dir, phase, None)

    def __getitem__(self, ind):
        """
        Retrieves the RGB and depth features for a given index.

        Args:
            ind (int): The index of the scene.

        Returns:
            dict: A dictionary containing the RGB and depth features, as well as their file paths.

        """
        rgb_feature_path = os.path.join(
            self.selected_scenes[ind]["dir"],
            "latent_rgb",
            str(self.selected_scenes[ind]["scene_id"]).zfill(6) + ".pt",
        )
        depth_feature_path = os.path.join(
            self.selected_scenes[ind]["dir"],
            "latent_depth",
            str(self.selected_scenes[ind]["scene_id"]).zfill(6) + ".pt",
        )

        rgb_feature = torch.load(rgb_feature_path, map_location="cpu")
        depth_feature = torch.load(depth_feature_path, map_location="cpu")
        rgb_feature.requires_grad = False
        depth_feature.requires_grad = False

        return {
            "rgb_feature": rgb_feature,
            "xyz_feature": depth_feature,
            "rgb_xyz_faeture": torch.cat([rgb_feature, depth_feature], dim=2),
            "rgb_feature_path": rgb_feature_path,
            "xyz_feature_path": depth_feature_path,
        }


class BOP_datamodule(pl.LightningDataModule):
    """
    LightningDataModule for BOP dataset.

    Args:
        root_dir (str): Root directory of the BOP dataset.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        obj_id_list (list): List of object IDs to include in the dataset.
        vis_ratio (float): Visibility ratio for the dataset.
        remove_repeated_objects (bool): Whether to remove repeated objects from the dataset.
    """

    def __init__(
        self, root_dir, batch_size, num_workers=0, transform=None, shuffle=True
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.shuffle = shuffle

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = BOP(self.root_dir, "train", transform=self.transform)
            self.val_dataset = BOP(self.root_dir, "val", transform=self.transform)
        if stage == "test":
            self.test_dataset = BOP(self.root_dir, "test", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


class BOP_feature_datamodule(pl.LightningDataModule):
    """
    LightningDataModule for BOP feature dataset.

    Args:
        root_dir (str): Root directory of the BOP dataset.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
    """

    def __init__(self, root_dir, batch_size, num_workers=0):
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = BOP_feature(self.root_dir, "train")
            self.val_dataset = BOP_feature(self.root_dir, "val")
        if stage == "test":
            self.test_dataset = BOP_feature(self.root_dir, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
