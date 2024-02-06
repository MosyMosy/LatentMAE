from typing import Any, List, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRSchedulerPLType
from models.ldm_autoencoder import AutoencoderKL
from models.mae_autoencoder import mae_vit_base_patch16, mae_vit_large_patch16
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from functools import reduce
import operator
from models.util import patchify, unpatchify
from enum import Enum

ddconfig = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
}


class adapter(nn.Linear):
    def __init__(
        self, in_patch_size: int, in_channel: int, out_patch_size: int, out_channel: int
    ) -> None:
        # in_len = reduce(operator.mul, in_shape, 1)
        # out_len = reduce(operator.mul, out_shape, 1)
        # assert in_len % num_patches + out_len % num_patches == 0
        in_features = in_patch_size**2 * in_channel
        out_features = out_patch_size**2 * out_channel
        super().__init__(in_features, out_features)

        self.in_patch_size = in_patch_size
        self.in_channel = in_channel
        self.out_patch_size = out_patch_size
        self.out_channel = out_channel

    def forward(self, input: Tensor) -> Tensor:
        B, C, H, W = input.shape
        assert int(C) == int(
            self.in_channel
        ), "The input channel size should be {}".format(self.in_channel)
        assert (
            H % self.in_patch_size == 0 or W % self.in_patch_size == 0
        ), "The input dimension is incorrect."

        out_H = int(H * self.out_patch_size / self.in_patch_size)
        out_W = int(W * self.out_patch_size / self.in_patch_size)
        input = patchify(input, patch_size=self.in_patch_size)
        input = input.reshape(-1, self.in_features)
        input = super().forward(input)
        input = input.reshape(
            -1, self.out_channel, self.out_patch_size, self.out_patch_size
        )
        input = unpatchify(
            input, (B, self.out_channel, out_H, out_W), patch_size=self.out_patch_size
        )
        return input


class AutoEncoder(pl.LightningModule):
    def __init__(
        self, pretrained_path="pretrained_ckpts/autoencoder_kl-f8.ckpt"
    ) -> None:
        super().__init__()
        self.pretrained_path = pretrained_path
        self.autoencoder = AutoencoderKL(embed_dim=4, ddconfig=ddconfig)

    def forward(self, x, latent_function=None, forward_mode="full"):
        B = x.shape[0]

        z = self.autoencoder.encode(x).sample()
        if forward_mode == "latent_feature":
            return z

        if latent_function is not None:
            z = latent_function(z)[0]

        z = self.autoencoder.decode(z)

        return z

    def load_pretrained_weights(self):
        self.autoencoder.init_from_ckpt(path=self.pretrained_path)
        print(f"AutoEncoder restored from {self.pretrained_path}")


class LatentMAE(pl.LightningModule):

    def __init__(
        self,
        latent_patch_size=14,
        latent_channel=4,
        mae_patch_size=16,
        mae_channel=3,
        pretrained_path="pretrained_ckpts/mae_pretrain_vit_base_full.pth",
        base_batch_size = 256,
    ) -> None:
        super().__init__()

        self.mae = mae_vit_base_patch16(norm_pix_loss=True)
        self.latent_patch_size = latent_patch_size

        self.in_adapter = adapter(
            latent_patch_size, latent_channel, mae_patch_size, mae_channel
        )
        self.out_adapter = adapter(
            mae_patch_size, mae_channel, latent_patch_size, latent_channel
        )

        self.pretrained_path = pretrained_path
        
        self.base_batch_size = base_batch_size

    def forward(self, x, mask_ratio=0.75, mae_forward_mode="full"):

        z = self.in_adapter(x)
        H, W = z.shape[2], z.shape[3]
        z = z.permute(0,2,3,1)
        z = torch.einsum('nhwc->nchw', z)
        z, mask, ids_restore = self.mae.forward_encoder(z, mask_ratio)

        if mae_forward_mode == "mae_feature":
            return z

        z = self.mae.forward_decoder(z, ids_restore)  # [N, L, p*p*3]
        z = self.mae.unpatchify(z, H=H, W=W)
        z = torch.einsum('nchw->nhwc', z)
 
        z = z.permute(0,3,1,2)
        z = self.out_adapter(z)  

        return z, mask, ids_restore
    
    def forward_loss(self, feature, pred, mask):
        """
        imgs: [N, 4, H, W]
        pred: [N, 4, H, W]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(feature)
        source = self.patchify(pred)
        if self.mae.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (source - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def patchify(self, feature):
        """
        feature: (N, 4, H, W)
        x: (N, L, patch_size**2 *4)
        """
        p = self.latent_patch_size
        assert feature.shape[2] % p == 0 and feature.shape[3] % p == 0

        h, w = feature.shape[2] // p , feature.shape[3] // p
        x = feature.reshape(shape=(feature.shape[0], 4, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(feature.shape[0], h * w, p**2 * 4))
        return x

    def load_pretrained_weights(self):
        sd = torch.load(self.pretrained_path, map_location="cpu")
        self.mae.load_state_dict(sd["model"], strict=True)
        print(f"MAE restored from {self.pretrained_path}")

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")
        
    def shared_step(self, batch, batch_idx, stage):
        x = batch["rgb_xyz_faeture"]
        pred, mask, _ = self.forward(x)
        loss = self.forward_loss(x, pred, mask)
        self.log(f"{stage}_loss", loss, batch_size=batch["rgb_xyz_faeture"].shape[0], sync_dist=True)
        return loss

        
    def configure_optimizers(self):  
        total_steps = self.trainer.estimated_stepping_batches
        devices, nodes = self.trainer.num_devices, self.trainer.num_nodes
        batch_size = self.trainer.train_dataloader.batch_size
        lr_scale = devices * nodes * batch_size / self.base_batch_size
        lr = 1e-3 * lr_scale

        optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=(.9, .95), weight_decay=0.05)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            cycle_momentum=False,
        )
        return {
            'optimizer': optim, 
            'lr_scheduler': {'scheduler': schedule, 'interval': 'step'}
        }
