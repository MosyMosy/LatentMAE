from models.ldm_autoencoder import AutoencoderKL
from models.mae_autoencoder import mae_vit_base_patch16
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from functools import reduce
import operator
from models.util import patchify, unpatchify

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
    def __init__(self, num_patches: int, in_shape: list, out_shape: list) -> None:
        in_len = reduce(operator.mul, in_shape, 1)
        out_len = reduce(operator.mul, out_shape, 1)
        assert in_len % num_patches + out_len % num_patches == 0
        in_features = int(in_len / num_patches)
        out_features = int(out_len / num_patches)
        super().__init__(in_features, out_features)

        self.num_patches = num_patches
        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(self, input: Tensor) -> Tensor:
        B = input.shape[0]
        out_patch_size = int(self.out_shape[1] / self.num_patches**0.5)
        input = input.reshape(B, *self.in_shape)
        input = patchify(input)
        input = input.reshape(B * self.num_patches, -1)
        input = super().forward(input)
        input = input.reshape(-1, self.out_shape[0], out_patch_size, out_patch_size)
        input = unpatchify(input, (B,) + tuple(self.out_shape), patch_size=out_patch_size)
        return input

class LatentMAE(pl.LightningModule):
    def __init__(self, latent_patch_size=14, input_shape=[3, 1568, 1568], latent_shape=[4, 196, 196], latent_mae_shape=[3, 224, 224]) -> None:
        super().__init__()
        assert latent_shape[1] % latent_patch_size == 0

        self.autoencoder = AutoencoderKL(embed_dim=4, ddconfig=ddconfig)
        self.mae = mae_vit_base_patch16()
        self.latent_patch_size = latent_patch_size
        self.num_patches = int(latent_shape[1]/latent_patch_size)**2
        self.input_shape = input_shape
        self.latent_shape = latent_shape
        self.latent_mae_shape = latent_mae_shape
        self.latent_mae_patch_size = int(
            latent_mae_shape[1] / (latent_shape[1]/latent_patch_size))
        # self.in_adaper = nn.Linear(self.latent_shape[0] * self.num_patches,
        #                            self.latent_mae_shape[0] * int(self.latent_mae_shape[2]**2 / self.num_patches))
        # self.out_adaper = nn.Linear(
        #     self.in_adaper.out_features, self.in_adaper.in_features)
        
        self.in_adaper = adapter(self.num_patches, latent_shape, latent_mae_shape)
        self.out_adaper = adapter(self.num_patches, latent_mae_shape, latent_shape)

    def forward(self, x, mask_ratio=0.75):
        assert list(x.shape)[1:] == self.input_shape
        B = x.shape[0]
        # Move the input to latent space
        z = self.autoencoder.encode(x).sample()
        z = self.in_adaper(z)
        z, mask, ids_restore = self.mae.forward_encoder(z, mask_ratio)
        z = self.mae.forward_decoder(z, ids_restore)  # [N, L, p*p*3]
        z = self.out_adaper(z)        
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
