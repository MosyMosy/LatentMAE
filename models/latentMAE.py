from models.ldm_autoencoder import AutoencoderKL
from mae.models_mae import mae_vit_base_patch16 
import pytorch_lightning as pl
import torch

ddconfig={"double_z": True,
      "z_channels": 4,
      "resolution": 256,
      "in_channels": 3,
      "out_ch": 3,
      "ch": 128,
      "ch_mult":[1,2,4,4],
      "num_res_blocks": 2,
      "attn_resolutions": [],
      "dropout": 0.0}


class LatentMAE(pl.LightningModule):
      def __init__(self) -> None:
            super().__init__()
            self.autoencoder = AutoencoderKL(embed_dim=4, ddconfig=ddconfig)
            self.mae = mae_vit_base_patch16()

      def forward(self, x):
            posterior = self.autoencoder.encode(x)
            z = posterior.sample()
            
            return self.autoencoder.decode(z)
            
      
      def load_pretrained_weights(self):
            self.autoencoder.init_from_ckpt(path="pretrained_ckpts/autoencoder_kl-f8.ckpt")
            
            sd = torch.load("pretrained_ckpts/mae_pretrain_vit_base.pth", map_location="cpu")
            self.mae.load_state_dict(sd["model"], strict=False)
            
