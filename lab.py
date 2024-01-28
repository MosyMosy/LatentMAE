import torch
from models.latentMAE import LatentMAE
from models.util import patchify, unpatchify

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
    resizer = transforms.Resize((224, 224))
    totensor = transforms.ToTensor()
    ToPIL = transforms.ToPILImage()
    sample = totensor(resizer(Image.open("lab/temp_data/sample.jpg"))).unsqueeze(0)    
    model = LatentMAE()
    model.load_pretrained_weights()
    reconstructed = model(sample, mode = LatentMAE.forward_mode.autoencoder)
    ToPIL(reconstructed[0]).save(os.path.join("lab/temp_data/reconstructed.jpg"))

test_model()
# test_autoencoder()

