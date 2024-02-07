import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from models.latentMAE import LatentMAE
from dataloaders.bop import BOP_feature_datamodule
from lightning.pytorch.cli import LightningCLI


def cli_main():
    cli = LightningCLI(LatentMAE, BOP_feature_datamodule, seed_everything_default=False)
    return cli

if __name__ == "__main__":
    cli_main()
