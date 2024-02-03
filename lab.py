import torch
from models.latentMAE import LatentMAE
from models.util import patchify, unpatchify

from dataloaders.bop import BOP_datamodule
from PIL import Image
from torchvision import transforms
import os


def test_model():
    model = LatentMAE()
    model.load_pretrained_weights()
    
    
    a = torch.range(0, 7375871)
    a = a.reshape(1,3,1568,1568)
    resutl = model(torch.rand(1,3,1568,1568))
    print(resutl.shape)  

def test_autoencoder():    
    resizer = transforms.Resize((448, 448))
    totensor = transforms.ToTensor()
    ToPIL = transforms.ToPILImage()
    sample = totensor(resizer(Image.open("lab/temp_data/sample.jpg"))).unsqueeze(0)    
    model = LatentMAE()
    model.load_pretrained_weights()
    reconstructed = model(sample, mode = LatentMAE.forward_mode.full)
    ToPIL(reconstructed[0]).save(os.path.join("lab/temp_data/reconstructed.jpg"))

def save_features():
    model = LatentMAE()
    model.to("cuda")
    model.load_pretrained_weights()
    
    # dataset = BOP("/export/livia/home/vision/Myazdanpanah/dataset/t-less", "train", obj_id_list=[1])
    data_loader = BOP_datamodule("/export/livia/home/vision/Myazdanpanah/dataset/t-less", batch_size=3)
    data_loader.setup(stage="fit")
    for b, sample in enumerate(data_loader.train_dataloader()):
        rgb = sample["rgb"].to("cuda")
        xyz_map = sample["xyz_map"].to("cuda")
        rgb_path = sample["rgb_path"]
        depth_path = sample["depth_path"]
        
        rec_rgb = model(rgb, mode = LatentMAE.forward_mode.latent_feature)
        rec_xyz_map = model(xyz_map, mode = LatentMAE.forward_mode.latent_feature)
        
        # save all the features in rec_rgb and rec_xyz_map  
        for i in range(rec_rgb.shape[0]):
            if_not_exists = lambda x: os.path.exists(x) == False
            if if_not_exists(os.path.dirname(rgb_path[i]).replace("rgb", "latent_rgb")):
                os.makedirs(os.path.dirname(rgb_path[i]).replace("rgb", "latent_rgb"))            
            torch.save(rec_rgb[i], rgb_path[i].replace("rgb", "latent_rgb").replace(".jpg", ".pt"))
            
            if if_not_exists(os.path.dirname(depth_path[i]).replace("depth", "latent_depth")):
                os.makedirs(os.path.dirname(depth_path[i]).replace("depth", "latent_depth"))
            torch.save(rec_xyz_map[i], depth_path[i].replace("depth", "latent_depth").replace(".png", ".pt"))
        print(f"Batch {b} done", end="\r")
        
# test_model()
test_autoencoder()
# save_features()

