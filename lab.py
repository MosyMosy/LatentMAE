import torch
import torch.nn as nn
from models.latentMAE import AutoEncoder, LatentMAE, ddconfig
from models.util import patchify, unpatchify

from dataloaders.bop import BOP_datamodule, BOP_feature
from PIL import Image
from torchvision import transforms
import os
from models.ldm_autoencoder import AutoencoderKL
import time


def test_model():
    model = LatentMAE()
    model.load_pretrained_weights()

    a = torch.range(0, 7375871)
    a = a.reshape(1, 3, 1568, 1568)
    resutl = model(torch.rand(1, 3, 1568, 1568))
    print(resutl.shape)


def test_autoencoder():
    resizer = transforms.Resize((448, 448))
    totensor = transforms.ToTensor()
    ToPIL = transforms.ToPILImage()
    sample = totensor(resizer(Image.open("lab/temp_data/sample.jpg"))).unsqueeze(0)
    model = AutoEncoder()
    model.load_pretrained_weights()

    # model = AutoencoderKL(embed_dim=4, ddconfig=ddconfig)
    # model.init_from_ckpt(path="pretrained_ckpts/autoencoder_kl-f8.ckpt")
    for param in model.parameters():
        param.requires_grad = False
    reconstructed = model(sample)
    ToPIL(reconstructed[0]).save(os.path.join("lab/temp_data/reconstructed.jpg"))


def test_latentmae():
    transform = transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ToPIL = transforms.ToPILImage()
    image = Image.open("lab/temp_data/sample2.jpg")
    sample = transform(image)
    autoencoder = AutoEncoder()
    autoencoder.load_pretrained_weights()

    latentmae = LatentMAE()
    latentmae.load_pretrained_weights()
    reconstructed = autoencoder(sample.unsqueeze(0), latent_function=latentmae)
    ToPIL(reconstructed[0]).save(os.path.join("lab/temp_data/reconstructed.jpg"))


def save_features(stage="test"):
    model = AutoEncoder()
    model.to("cuda:0")
    model.load_pretrained_weights()
    model.eval()

    transform = transforms.Compose(
                [
                    transforms.Resize((1568, 1568)),
                    transforms.ToTensor(),                    
                ]
            )
    data_loader = BOP_datamodule(
        "/export/livia/home/vision/Myazdanpanah/dataset/t-less", batch_size=1, transform=transform, shuffle=False
    )
    if stage == "test":
        data_loader.setup(stage="test")
        loader = data_loader.test_dataloader()
    else:
        data_loader.setup(stage="fit")
        loader = data_loader.train_dataloader()
    for b, sample in enumerate(loader):
        rgb = sample["rgb"].to("cuda:0")
        xyz_map = sample["xyz_map"].to("cuda:0")
        rgb_path = sample["rgb_path"]
        depth_path = sample["depth_path"]
        with torch.no_grad():
            rec_rgb = model(rgb, forward_mode="latent_feature")
            rec_xyz_map = model(xyz_map, forward_mode="latent_feature")

        # save all the features in rec_rgb and rec_xyz_map
        for i in range(rec_rgb.shape[0]):
            if_not_exists = lambda x: os.path.exists(x) == False
            if if_not_exists(os.path.dirname(rgb_path[i]).replace("rgb", "latent_large_rgb")):
                os.makedirs(os.path.dirname(rgb_path[i]).replace("rgb", "latent_large_rgb"))
            torch.save(
                rec_rgb[i],
                rgb_path[i].replace("rgb", "latent_large_rgb").replace(".jpg", ".pt").replace(".png", ".pt")
            )

            if if_not_exists(
                os.path.dirname(depth_path[i]).replace("depth", "latent_large_depth")
            ):
                os.makedirs(
                    os.path.dirname(depth_path[i]).replace("depth", "latent_large_depth")
                )
            torch.save(
                rec_xyz_map[i],
                depth_path[i].replace("depth", "latent_large_depth").replace(".png", ".pt"),
            )
        print(f"Batch {b} done", end="\r")
    print(f"stage {stage} done")


def test_AE_size():
    model = AutoEncoder()
    model.to("cuda")
    model.load_pretrained_weights()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    data_loader = BOP_datamodule(
        "/export/livia/home/vision/Myazdanpanah/dataset/t-less", batch_size=8
    )
    data_loader.setup(stage="test")
    loader = data_loader.test_dataloader()
    for b, sample in enumerate(loader):
        rgb = sample["rgb"].to("cuda")
        xyz_map = sample["xyz_map"].to("cuda")
        rgb_path = sample["rgb_path"]
        depth_path = sample["depth_path"]
        # with torch.no_grad():
        rec_rgb = model(rgb, forward_mode="Full")
        rec_xyz_map = model(xyz_map, forward_mode="Full")


def test_AE_size_fake():
    model = AutoEncoder()
    model.to("cuda")
    model.load_pretrained_weights()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    totaltime = 0
    batch = torch.rand(100, 1, 3, 1568, 1568)
    for i, sample in enumerate(batch):
        start_time = time.time()
        _ = model(sample.to("cuda"), forward_mode="Full")
        end_time = time.time()
        duration = end_time - start_time
        totaltime += duration
        print(f"Iteration {i}: {end_time - start_time} seconds", end="\r")
    print(f"Average time: {totaltime / len(batch)} seconds")


def test_AE_rec_loss():
    model = AutoEncoder()
    model.to("cuda")
    model.load_pretrained_weights()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    data_loader = BOP_datamodule(
        "/export/livia/home/vision/Myazdanpanah/dataset/t-less", batch_size=10
    )
    data_loader.setup(stage="test")
    loader = data_loader.test_dataloader()
    total_loss = 0
    for b, sample in enumerate(loader):
        rgb_xyz = sample["rgb_xyz"].to("cuda")
        rgb_xyz_rec = model(rgb_xyz, forward_mode="Full")
        total_loss += nn.MSELoss()(rgb_xyz, rgb_xyz_rec).item()
        print(f"Batch {b} done", end="\r")

    print(f"Average loss: {total_loss / len(data_loader.test_dataset)}")


def test_autoencoder_rec(
    checkpoint_path="lightning_logs/checkpoints/LatentMAE_AE/v_frozen_ae/LatentMAE_epoch=51_step=433368_val_loss=0.7501.ckpt",
):
    transform = transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ]
    )
    data_module = BOP_datamodule(
        "/export/livia/home/vision/Myazdanpanah/dataset/t-less",
        batch_size=1,
        transform=transform,
    )
    data_module.setup(stage="fit")
    dataset = data_module.train_dataset
    mae_func = LatentMAE.load_from_checkpoint(checkpoint_path, map_location="cpu")
    mae_func.eval()
    model = AutoEncoder()
    model.load_pretrained_weights()

    for i in range(10):
        rand_indes = torch.randint(0, len(dataset), (1,))[0].item()
        random_sample = dataset[rand_indes]["rgb"].unsqueeze(0)
        with torch.no_grad():
            reconstructed_mae = model(random_sample, latent_function=mae_func)
            reconstructed_ae = model(random_sample)
        transforms.ToPILImage()(reconstructed_mae[0]).save(
            "lab/temp_data/rec_mae_full/rec_{}.jpg".format(rand_indes)
        )
        transforms.ToPILImage()(reconstructed_ae[0]).save(
            "lab/temp_data/rec_mae_full/rec_ae_{}.jpg".format(rand_indes)
        )
        transforms.ToPILImage()(random_sample[0]).save(
            "lab/temp_data/rec_mae_full/org_{}.jpg".format(rand_indes)
        )
        print(f"Image {i} done", end="\r")
    print("Done")


# test_model()
# test_autoencoder()

save_features(stage="test")
save_features(stage="fit")


# test_AE_size()
# test_AE_size_fake()
# test_latentmae()

# test_AE_rec_loss()

# test_autoencoder_rec()
