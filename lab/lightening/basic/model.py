from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

    
class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def generic_step(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
        
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss = self.generic_step(batch=batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.generic_step(batch=batch)
        self.log("valid_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.generic_step(batch=batch)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer