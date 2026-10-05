"""
Lightning tasks for FluCoMa MLPs
"""

from typing import List, Literal

import lightning as L
from sklearn.metrics import accuracy_score
import torch

from pydataknot.model import FluidMLP


class FluidMLPRegressor(L.LightningModule):
    """
    A PyTorch Lightning module for training and evaluation of a FluCoMa MLP
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        activation: int,
        output_activation: int,
        learn_rate: float = 1e-3,
        max_iter: int = 1000,
        validation: float = 0.2,
        batch_size: int = 32,
        momentum: float = 0.9,
        optimizer: Literal["sgd", "adam"] = "sgd",
    ):
        super().__init__()
        self.model = FluidMLP(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            activation=activation,
            output_activation=output_activation,
        )
        self.learn_rate = learn_rate
        self.max_iter = max_iter
        self.validation = validation
        self.batch_size = batch_size
        self.momentum = momentum
        self.optimizer_ = optimizer
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        """
        x, y = batch

        # Forward pass
        y_hat = self(x)

        # Compute loss
        loss = self.loss_function(y_hat, y)

        # Log the loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        """
        x, y = batch

        # Forward pass
        y_hat = self.model(x)

        # Compute loss
        loss = self.loss_function(y_hat, y)

        # Log the validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        """
        if self.optimizer_ == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.learn_rate,
                momentum=self.momentum,
            )
        elif self.optimizer_ == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.learn_rate,
            )
        else:
            raise ValueError(
                f"Unknown optimizer {self.optimizer_}, must be 'sgd' or 'adam'"
            )


class FluidMLPClassifier(FluidMLPRegressor):
    """
    A PyTorch Lightning module for training and evaluation of a FluCoMa MLP Classifier.
    This is almost identical to the regressor, but the output activation is always
    a sigmoid.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        activation: int,
        learn_rate: float = 1e-3,
        max_iter: int = 1000,
        validation: float = 0.2,
        batch_size: int = 32,
        momentum: float = 0.9,
        optimizer: Literal["sgd", "adam"] = "sgd",
        loss_fn: Literal["mse", "bce"] = "mse",
    ):
        super().__init__(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            activation=activation,
            output_activation=1,
            learn_rate=learn_rate,
            max_iter=max_iter,
            validation=validation,
            batch_size=batch_size,
            momentum=momentum,
            optimizer=optimizer,
        )
        if loss_fn == "mse":
            self.loss_function = torch.nn.MSELoss()
        elif loss_fn == "bce":
            self.loss_function = torch.nn.BCELoss()
        else:
            raise ValueError(
                f"Unsupported loss function: {loss_fn}. Must be one of 'mse' or 'bce'"
            )

    def validation_step(self, batch, batch_idx):
        """
        Override base validation to include accuracy reporting
        """
        x, y = batch

        y_hat = self(x)

        # Compute loss
        loss = self.loss_function(y_hat, y)

        # Compute accuracy
        y_hat = torch.argmax(y_hat, dim=-1, keepdim=False)
        y = torch.argmax(y, dim=-1)
        acc = accuracy_score(y_hat.cpu().numpy(), y.cpu().numpy(), normalize=True)

        # Log the validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
