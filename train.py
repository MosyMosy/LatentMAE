import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.latentMAE import LatentMAE
from dataloaders.bop import BOP_feature_datamodule


batch_size = 4
# Define your data module
data_module = BOP_feature_datamodule(
    "/export/livia/home/vision/Myazdanpanah/dataset/t-less", batch_size, 0
)  # replace with your actual data module

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
    devices=1,
    strategy="auto",
    log_every_n_steps=1,
    max_epochs=100,
    logger=logger,
    callbacks=[checkpoint_callback],
)

# Train the model
trainer.fit(model, data_module)
