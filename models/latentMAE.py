from models.ldm_autoencoder import AutoencoderKL
from models.mae_autoencoder import mae_vit_base_patch16
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
    def __init__(self, in_patch_size: int, in_channel: int, out_patch_size: int, out_channel: int) -> None:
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
            self.in_channel), "The input channel size should be {}".format(self.in_channel)
        assert H % self.in_patch_size == 0 or W % self.in_patch_size == 0, "The input dimension is incorrect."

        out_H = int(H * self.out_patch_size / self.in_patch_size)
        out_W = int(W * self.out_patch_size / self.in_patch_size)
        input = patchify(input, patch_size=self.in_patch_size)
        input = input.reshape(-1, self.in_features)
        input = super().forward(input)
        input = input.reshape(-1, self.out_channel,
                              self.out_patch_size, self.out_patch_size)
        input = unpatchify(input, (B, self.out_channel, out_H,
                           out_W), patch_size=self.out_patch_size)
        return input


class LatentMAE(pl.LightningModule):
    class forward_mode(Enum):
        full = 0
        latent_feature = 1
        mea_feature = 2
        autoencoder = 3

    def __init__(self, latent_patch_size=14, latent_channel=4, mae_patch_size=16, mae_channel=3) -> None:
        super().__init__()

        self.autoencoder = AutoencoderKL(embed_dim=4, ddconfig=ddconfig)
        self.mae = mae_vit_base_patch16()
        self.latent_patch_size = latent_patch_size

        self.in_adapter = adapter(
            latent_patch_size, latent_channel, mae_patch_size, mae_channel)
        self.out_adapter = adapter(
            mae_patch_size, mae_channel, latent_patch_size, latent_channel)


    def forward(self, x, mask_ratio=0.75, mode: forward_mode = forward_mode.full):
        B = x.shape[0]
        z = self.autoencoder.encode(x).sample()
        
        if mode == LatentMAE.forward_mode.latent_feature:
            return z

        if mode != LatentMAE.forward_mode.autoencoder:
            z = self.in_adapter(z)
            mae_shape = z.shape
            z, mask, ids_restore = self.mae.forward_encoder(z, mask_ratio)
            if mode == LatentMAE.forward_mode.mea_feature:
                return z

            z = self.mae.forward_decoder(z, ids_restore)  # [N, L, p*p*3]
            z = self.mae.unpatchify(z)
            z = self.out_adapter(z)

        z = self.autoencoder.decode(z)

        return z

    def load_pretrained_weights(self):
        path = "pretrained_ckpts/autoencoder_kl-f8.ckpt"
        self.autoencoder.init_from_ckpt(
            path=path)
        print(f"AutoEncoder restored from {path}")

        path = "pretrained_ckpts/mae_pretrain_vit_base_full.pth"
        sd = torch.load(path, map_location="cpu"
                        )
        self.mae.load_state_dict(sd["model"], strict=True)
        print(f"MAE restored from {path}")
