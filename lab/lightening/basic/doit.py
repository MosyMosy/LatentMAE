import os
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
import torch.utils.data as data


from model import LitAutoEncoder


transform = transforms.ToTensor()
dataset = MNIST(os.getcwd(), download=True, transform=transform)
test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)

train_set_size = int(len(dataset) * 0.8)
valid_set_size = len(dataset) - train_set_size
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(
    dataset, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(train_set, batch_size=256)
valid_loader = DataLoader(valid_set, batch_size=256)
test_loader = DataLoader(test_set, batch_size=256)
# model
autoencoder = LitAutoEncoder()

# train model
trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=2)
trainer.fit(model=autoencoder, train_dataloaders=train_loader,
            val_dataloaders=valid_loader)
trainer.test(autoencoder, dataloaders=test_loader)
