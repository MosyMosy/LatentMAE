import logging
import os
import json

# import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import ToPILImage, ToTensor, Resize, Compose


from PIL import Image
import open3d as o3d


class BOP(Dataset):

    def __init__(self, root_dir, phase, transform=None, obj_id_list=[], vis_ratio=0, remove_repeated_objects=False):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        if len(obj_id_list) == 0:
            obj_id_list = range(1, 31)

        self.root = None
        if isinstance(root_dir, str):
            if os.path.exists(f'{root_dir}'):
                self.root = root_dir

        if self.root is None:
            raise AssertionError(f'Dataset not found in {root_dir}')
        else:
            self.logger.info(f'Loading data from {self.root}')

        # set the directories related to the T-Less dataset
        self.train_dir = os.path.join(self.root, "train_pbr")
        self.test_dir = os.path.join(
            self.root, "tless_test_primesense_bop19", "test_primesense")
        self.cad_dir = os.path.join(
            self.root, "tless_models", "models_cad")

        # each directory is related to a specific camera and scene setup
        self.selected_scenes = []
        assert phase in ['train', 'val', 'test']
        base_dir = self.train_dir if phase in ['train'] else self.test_dir
        setup_dirs = [name for name in os.listdir(
            base_dir) if os.path.isdir(os.path.join(base_dir))]
        scene_obj_list = []
        for setup_dir in setup_dirs:
            scene_gt_path = os.path.join(base_dir, setup_dir, "scene_gt.json")
            camera_gt_path = os.path.join(
                base_dir, setup_dir, "scene_camera.json")
            scene_info_path = os.path.join(
                base_dir, setup_dir, "scene_gt_info.json")
            with open(scene_gt_path) as scenes_f, open(camera_gt_path) as camera_f, open(scene_info_path) as scene_info_f:
                scenes_dic = json.load(scenes_f)
                cameras_dic = json.load(camera_f)
                scene_info_dic = json.load(scene_info_f)
                for scene_id, obj_list in scenes_dic.items():
                    for i, obj_dic in enumerate(obj_list):
                        for obj_id in obj_id_list:
                            if (obj_dic["obj_id"] == obj_id):
                                # remove low visible items
                                if scene_info_dic[scene_id][i]["visib_fract"] < vis_ratio:
                                    continue
                                data_dic = {"dir": os.path.join(
                                    base_dir, setup_dir), "scene_id": int(scene_id), "obj_index": i}
                                data_dic.update(obj_dic)
                                data_dic.update(cameras_dic[scene_id])
                                self.selected_scenes.append(data_dic)
                                scene_obj_list.append(
                                    [int(setup_dir), int(scene_id), obj_id])

        if len(self.selected_scenes) == 0:
            raise AssertionError(
                "there is no selected obj_num {0} in the given directory.".format(obj_id_list))

        # remove the scenes with repeated objects
        if remove_repeated_objects:
            _, indexes, count = np.unique(
                np.array(scene_obj_list), axis=0, return_counts=True, return_index=True)
            indexes = indexes[count == 1]
            self.selected_scenes = [self.selected_scenes[i] for i in indexes]

        if transform is None:
            self.transform = Compose([ToTensor()])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.selected_scenes)

    def __getitem__(self, ind):
        rgb_path = os.path.join(self.selected_scenes[ind]["dir"], "rgb", str(
            self.selected_scenes[ind]["scene_id"]).zfill(6) + ".jpg")
        depth_path = os.path.join(self.selected_scenes[ind]["dir"], "depth", str(
            self.selected_scenes[ind]["scene_id"]).zfill(6) + ".png")
        
        rgb = Image.open(rgb_path)
        rgb = pil_to_tensor(rgb)
        
        cam_K = np.array(self.selected_scenes[ind]["cam_K"]).reshape(3, 3)
        cam_depth_scale = 1 / self.selected_scenes[ind]["depth_scale"]

        depth_im = o3d.io.read_image(depth_path)
        im_width, im_height = depth_im.get_max_bound()
        pinhole_cam = o3d.camera.PinholeCameraIntrinsic(
            int(im_width), int(im_height), cam_K)
        scene_pcl = o3d.geometry.PointCloud.create_from_depth_image(
            depth_im, pinhole_cam, depth_scale=cam_depth_scale, depth_trunc=float('inf'))
        scene_pcl = scene_pcl.points
        xyz = torch.from_numpy(np.array(scene_pcl.points, dtype=np.float32))
        xyz = xyz.reshape(*rgb.shape)
        
        # normalize xyz to [0, 255]
        xyz = (xyz - xyz.min()) / xyz.max() * 255
        
        return {"rgb": self.transform(rgb), "depth": self.transform(xyz), "rgb_path": rgb_path, "depth_path": depth_path}
