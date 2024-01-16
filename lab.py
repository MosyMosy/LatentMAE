import torch
from models.latentMAE import LatentMAE


def test_model():
    model = LatentMAE()
    model.load_pretrained_weights()
    resutl = model(torch.rand(1,3,512,512))
    print(resutl.shape)  
    
test_model()