import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from models.latentMAE import LatentMAE
from dataloaders.bop import BOP_feature_datamodule
from lightning.pytorch.cli import LightningCLI
from models.mae_autoencoder import mae_vit_base_patch16, mae_vit_large_patch16
import os.path as Path 


def cli_main():    
    cli = LightningCLI(LatentMAE, BOP_feature_datamodule)

    # from https://github.com/Lightning-AI/pytorch-lightning/issues/1207
    csv_logger = CSVLogger("logs", name="LatentMAE")
    tb_logger = TensorBoardLogger("logs", name="LatentMAE")
    checkpoint_dir = (
        Path(tb_logger.save_dir)
        / tb_logger.experiment.name
        / f"version_{tb_logger.experiment.version}"
        / "checkpoints"
    )
    filepath = checkpoint_dir / "{epoch}-{val_loss:.4f}"
    checkpoint_cb = ModelCheckpoint(filepath=str(filepath),monitor="val_loss", save_top_k=3, mode="min", save_last=True, verbose=True)

    cli.trainer.callbacks = [checkpoint_cb, csv_logger]
    cli.trainer.logger = tb_logger
    
    return cli





def main():
    batch_size = 256
    # Define your data module
    data_module = BOP_feature_datamodule(
        root_dir="/export/livia/home/vision/Myazdanpanah/dataset/t-less", batch_size= batch_size, num_workers=32
    ) 

    # Define your model
    model = LatentMAE(base_batch_size=batch_size)  
    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="latentmae-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # Define TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="LatentMAE")

    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[2,3],
        strategy="DDP",
        log_every_n_steps=1,
        max_epochs=500,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, data_module)


if __name__ == '__main__':
    cli_main()