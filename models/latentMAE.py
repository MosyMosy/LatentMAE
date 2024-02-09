from typing import Any, Dict, List, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRSchedulerPLType
from models.ldm_autoencoder import AutoencoderKL
from models.mae_autoencoder import mae_vit_base_patch16, mae_vit_large_patch16
import lightning.pytorch as pl
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
    """
    Adapter class for transforming input patch tensors using linear transformation.

    Args:
        in_patch_size (int): Size of the input patch.
        in_channel (int): Number of input channels.
        out_patch_size (int): Size of the output patch.
        out_channel (int): Number of output channels.
    """

    def __init__(
        self, in_patch_size: int, in_channel: int, out_patch_size: int, out_channel: int
    ) -> None:
        """
        Initializes the LatentMAE model.

        Args:
            in_patch_size (int): The size of the input patch.
            in_channel (int): The number of input channels.
            out_patch_size (int): The size of the output patch.
            out_channel (int): The number of output channels.
        """
        in_features = in_patch_size**2 * in_channel
        out_features = out_patch_size**2 * out_channel
        super().__init__(in_features, out_features)

        self.in_patch_size = in_patch_size
        self.in_channel = in_channel
        self.out_patch_size = out_patch_size
        self.out_channel = out_channel

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass of the Adapter model. This pathifies the input tensor, applies the linear transformation, and unpatchifies the output tensor.

        Args:
            input (Tensor): The input tensor of shape (B, C, H, W).

        Returns:
            Tensor: The output tensor of shape (B, self.out_channel, out_H, out_W).
        """
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
    """
    LatentMAE is a PyTorch Lightning module that implements a latent space
    autoencoder for image reconstruction.

    Args:
        latent_patch_size (int): The patch size of the latent space. Default is 14.
        latent_channel (int): The number of channels in the latent space. Default is 4.
        mae_patch_size (int): The patch size of the MAE (Multi-Adversarial Encoder) model. Default is 16.
        mae_channel (int): The number of channels in the MAE model. Default is 3.
        pretrained_path (str): The path to the pretrained weights of the MAE model. Default is "pretrained_ckpts/mae_pretrain_vit_base_full.pth".
        mae_backbone (nn.Module): The backbone model of the MAE. Default is mae_vit_base_patch16.
        mae_norm_pix_loss (bool): Whether to normalize the pixel loss in the MAE model. Default is True.
        base_batch_size (int): The base batch size for training. Default is 256.
        use_pretrained (bool): Whether to use the pretrained weights for the MAE model. Default is True.
    """

    def __init__(
        self,
        latent_patch_size=14,
        latent_channel=4,
        mae_patch_size=16,
        mae_channel=3,
        pretrained_path="pretrained_ckpts/mae_pretrain_vit_base_full.pth",
        mae_backbone="mae_vit_base_patch16",
        mae_norm_pix_loss=True,
        base_batch_size=256,
        use_pretrained=True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if mae_backbone == "mae_vit_base_patch16":
            self.mae = mae_vit_base_patch16(norm_pix_loss=mae_norm_pix_loss)
        elif mae_backbone == "mae_vit_large_patch16":
            self.mae = mae_vit_large_patch16(norm_pix_loss=mae_norm_pix_loss)
        else:
            raise ValueError(f"Unknown backbone: {mae_backbone}")

        self.latent_patch_size = latent_patch_size

        self.in_adapter = adapter(
            latent_patch_size, latent_channel, mae_patch_size, mae_channel
        )
        self.out_adapter = adapter(
            mae_patch_size, mae_channel, latent_patch_size, latent_channel
        )

        self.pretrained_path = pretrained_path

        self.base_batch_size = base_batch_size

        self.use_pretrained = use_pretrained

    def setup(self, stage: Union[str, None]) -> None:
        if self.use_pretrained:
            sd = torch.load(self.pretrained_path, map_location="cpu")
            self.mae.load_state_dict(sd["model"], strict=True)
            self.mae.eval
            print(f"MAE restored from {self.pretrained_path}")

    def forward(self, x, mask_ratio=0.75, mae_forward_mode="full"):
        """
        Forward pass of the LatentMAE model.

        Args:
            x (torch.Tensor): The input tensor of shape (N, C, H, W).
            mask_ratio (float): The ratio of patches to be masked during encoding. Default is 0.75.
            mae_forward_mode (str): The forward mode of the MAE model. Default is "full".

        Returns:
            torch.Tensor: The output tensor of shape (N, C, H, W).
            torch.Tensor: The mask tensor of shape (N, L).
            torch.Tensor: The restored IDs tensor of shape (N, L).
        """
        z = self.in_adapter(x)
        H, W = z.shape[2], z.shape[3]
        z = z.permute(0, 2, 3, 1)
        z = torch.einsum("nhwc->nchw", z)
        z, mask, ids_restore = self.mae.forward_encoder(z, mask_ratio)

        if mae_forward_mode == "mae_feature":
            return z

        z = self.mae.forward_decoder(z, ids_restore)  # [N, L, p*p*3]
        z = self.mae.unpatchify(z, H=H, W=W)
        z = torch.einsum("nchw->nhwc", z)

        z = z.permute(0, 3, 1, 2)
        z = self.out_adapter(z)

        return z, mask, ids_restore

    def forward_loss(self, feature, pred, mask):
        """
        Compute the forward loss of the LatentMAE model.

        Args:
            feature (torch.Tensor): The input feature tensor of shape (N, C, H, W).
            pred (torch.Tensor): The predicted tensor of shape (N, C, H, W).
            mask (torch.Tensor): The mask tensor of shape (N, L), where 0 is keep and 1 is remove.

        Returns:
            torch.Tensor: The computed loss value.
        """
        target = self.patchify(feature)
        source = self.patchify(pred)
        if self.mae.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (source - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def patchify(self, feature):
        """
        Patchify the feature tensor.

        Args:
            feature (torch.Tensor): The input feature tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: The patchified tensor of shape (N, L, patch_size**2 * 4).
        """
        p = self.latent_patch_size
        assert feature.shape[2] % p == 0 and feature.shape[3] % p == 0

        h, w = feature.shape[2] // p, feature.shape[3] // p
        x = feature.reshape(shape=(feature.shape[0], 4, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(feature.shape[0], h * w, p**2 * 4))
        return x

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
        self.log(
            f"{stage}_loss",
            loss,
            batch_size=batch["rgb_xyz_faeture"].shape[0],
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches
        devices, nodes = self.trainer.num_devices, self.trainer.num_nodes
        batch_size = self.trainer.train_dataloader.batch_size
        lr_scale = devices * nodes * batch_size / self.base_batch_size
        lr = 1e-3 * lr_scale

        optim = torch.optim.AdamW(
            self.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05
        )
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            cycle_momentum=False,
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": schedule, "interval": "step"},
        }


feature_autoencoder = AutoencoderKL(embed_dim=4, ddconfig=ddconfig)


class LatentMAE_AE(pl.LightningModule):
    """
    LatentMAE_AE is a PyTorch Lightning module that implements a latent space autoencoder with a Masked Autoencoder (MAE) backbone.
    It is used for image reconstruction and feature extraction tasks.

    Args:
        latent_patch_size (int): The patch size of the latent space. Default is 14.
        latent_channel (int): The number of channels in the latent space. Default is 4.
        mae_patch_size (int): The patch size of the Masked Autoencoder (MAE). Default is 16.
        mae_channel (int): The number of channels in the Masked Autoencoder (MAE). Default is 3.
        image_patch_size (int): The patch size of the input image. Default is 112.
        pretrained_path (str): The path to the pretrained checkpoint for the MAE backbone. Default is "pretrained_ckpts/mae_pretrain_vit_base_full.pth".
        mae_backbone (str): The backbone architecture of the Masked Autoencoder (MAE). Default is "mae_vit_base_patch16".
        mae_norm_pix_loss (bool): Whether to use normalized pixel loss in the MAE. Default is True.
        base_batch_size (int): The base batch size for training. Default is 256.
        use_initial_pretrained (bool): Whether to use the initial pretrained weights for the MAE backbone. Default is True.
        AE_pretrained_path (str): The path to the pretrained checkpoint for the feature autoencoder. Default is "pretrained_ckpts/autoencoder_kl-f8.ckpt".
    """

    def __init__(
        self,
        latent_patch_size=14,
        latent_channel=4,
        mae_patch_size=16,
        mae_channel=3,
        pretrained_path="pretrained_ckpts/mae_pretrain_vit_base_full.pth",
        mae_backbone="mae_vit_base_patch16",
        mae_norm_pix_loss=True,
        base_batch_size=256,
        use_initial_pretrained=True,
        AE_pretrained_path="pretrained_ckpts/autoencoder_kl-f8.ckpt",
    ) -> None:
        """
        Initializes the LatentMAE model.

        Args:
            latent_patch_size (int): The size of the latent patches.
            latent_channel (int): The number of channels in the latent representation.
            mae_patch_size (int): The size of the MAE patches.
            mae_channel (int): The number of channels in the MAE representation.
            pretrained_path (str): The path to the pretrained checkpoint.
            mae_backbone (str): The backbone architecture for the MAE model.
            mae_norm_pix_loss (bool): Whether to normalize the pixel loss in the MAE model.
            base_batch_size (int): The base batch size for training.
            use_initial_pretrained (bool): Whether to use the initial pretrained weights.
            AE_pretrained_path (str): The path to the pretrained autoencoder checkpoint.
        """
        super().__init__()
        self.save_hyperparameters()
        if mae_backbone == "mae_vit_base_patch16":
            self.mae = mae_vit_base_patch16(norm_pix_loss=mae_norm_pix_loss)
        elif mae_backbone == "mae_vit_large_patch16":
            self.mae = mae_vit_large_patch16(norm_pix_loss=mae_norm_pix_loss)
        else:
            raise ValueError(f"Unknown backbone: {mae_backbone}")

        self.AE_pretrained_path = AE_pretrained_path

        self.latent_patch_size = latent_patch_size
        self.in_adapter = adapter(
            latent_patch_size, latent_channel, mae_patch_size, mae_channel
        )
        self.out_adapter = adapter(
            mae_patch_size, mae_channel, latent_patch_size, latent_channel
        )
        self.pretrained_path = pretrained_path
        self.base_batch_size = base_batch_size
        self.use_initial_pretrained = use_initial_pretrained

    def setup(self, stage: Union[str, None]) -> None:
        """
        Setup method to load the pretrained weights for the MAE backbone and the feature autoencoder.

        Args:
            stage (str, None): The current stage of training. Default is None.
        """
        if self.use_initial_pretrained:
            sd = torch.load(self.pretrained_path, map_location="cpu")
            self.mae.load_state_dict(sd["model"], strict=True)
            self.mae.eval
            print(f"MAE restored from {self.pretrained_path}")

        feature_autoencoder.init_from_ckpt(path=self.AE_pretrained_path)
        feature_autoencoder.eval()
        feature_autoencoder.to(self.device)
        print(f"AutoEncoder restored from {self.AE_pretrained_path}")

    def forward(self, x, mask_ratio=0.75, mae_forward_mode="full"):
        """
        Forward pass of the LatentMAE model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            mask_ratio (float): Ratio of masked values in the latent space. Default is 0.75.
            mae_forward_mode (str): Forward mode for the MAE model. Options are "full" and "mae_feature". Default is "full".

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
            torch.Tensor: Mask tensor of shape (N, C, H, W).
            torch.Tensor: Restored IDs tensor of shape (N, C, H, W).
        """
        feature_autoencoder.eval()

        with torch.no_grad():
            z = feature_autoencoder.encode(x).sample()
        z.requires_grad = True

        z = self.in_adapter(z)
        H, W = z.shape[2], z.shape[3]
        z = z.permute(0, 2, 3, 1)
        z = torch.einsum("nhwc->nchw", z)
        z, mask, ids_restore = self.mae.forward_encoder(z, mask_ratio)

        if mae_forward_mode == "mae_feature":
            return z

        z = self.mae.forward_decoder(z, ids_restore)  # [N, L, p*p*3]
        z = self.mae.unpatchify(z, H=H, W=W)
        z = torch.einsum("nchw->nhwc", z)

        z = z.permute(0, 3, 1, 2)
        z = self.out_adapter(z)

        with torch.no_grad():
            z = feature_autoencoder.decode(z)
        z.requires_grad = True

        return z, mask, ids_restore

    def forward_loss(self, image, pred, mask):
        """
        Calculates the pixel level MSE loss between the input image and the predicted image.

        Args:
            image (torch.Tensor): The input image.
            pred (torch.Tensor): The predicted image.
            mask (torch.Tensor): The mask indicating the valid regions of the image.

        Returns:
            torch.Tensor: The forward loss between the input image and the predicted image.
        """
        loss = nn.MSELoss()(image, pred)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step of the LatentMAE_AE module.

        Args:
            batch (Any): The input batch.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the LatentMAE_AE module.

        Args:
            batch (Any): The input batch.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """
        Test step of the LatentMAE_AE module.

        Args:
            batch (Any): The input batch.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "test")

    def shared_step(self, batch, batch_idx, stage):
        """
        Shared step for training, validation, and testing.

        Args:
            batch (Any): The input batch.
            batch_idx (int): The index of the current batch.
            stage (str): The current stage of training.

        Returns:
            torch.Tensor: The loss value.
        """
        x = batch["rgb_xyz"]
        pred, mask, _ = self.forward(x)
        loss = self.forward_loss(x, pred, mask)
        self.log(
            f"{stage}_loss",
            loss,
            batch_size=batch["rgb_xyz"].shape[0],
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: The optimizer and learning rate scheduler.
        """
        total_steps = self.trainer.estimated_stepping_batches
        devices, nodes = self.trainer.num_devices, self.trainer.num_nodes
        batch_size = self.trainer.train_dataloader.batch_size
        lr_scale = devices * nodes * batch_size / self.base_batch_size
        lr = 1e-3 * lr_scale

        optim = torch.optim.AdamW(
            self.mae.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05
        )
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            cycle_momentum=False,
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": schedule, "interval": "step"},
        }
