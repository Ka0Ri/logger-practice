from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger

neptune_logger = NeptuneLogger(
    api_key="ANONYMOUS",  # replace with your own
    project="practice/pytorch-lightning-integration",  # format "<WORKSPACE/PROJECT>"
    tags=["training", "resnet"],  # optional
)

trainer = Trainer(max_epochs=10, logger=neptune_logger)

from neptune.new.types import File
from pytorch_lightning import LightningModule


class LitModel(LightningModule):
    def training_step(self, batch, batch_idx):
        # log metrics
        acc = ...
        self.log("train/loss", loss)

    def any_lightning_module_function_or_hook(self):
        # log images
        img = ...
        self.logger.experiment["train/misclassified_images"].log(File.as_image(img))

        # generic recipe
        metadata = ...
        self.logger.experiment["your/metadata/structure"].log(metadata)