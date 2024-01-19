import torch
from models.latentMAE import LatentMAE
from models.util import patchify, unpatchify


def test_model():
    model = LatentMAE()
    model.load_pretrained_weights()
    resutl = model(torch.rand(1,3,1568,1568))
    print(resutl.shape)  
    
test_model()


