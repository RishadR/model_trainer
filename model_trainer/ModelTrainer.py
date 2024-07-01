"""
A model trainer module used to train and validate a model. This module is designed to be used with PyTorch models and
uses a Composer design pattern to allow for easy customization of the training process.
"""
from typing import Dict, Optional, Tuple, Type
from torch import nn
import torch
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from .misc import DATA_LOADER_INPUT_INDEX
from .DataLoaderGenerators import DataLoaderGenerator
from .validation_methods import ValidationMethod
from .loss_funcs import LossFunction
from .early_stopping import EarlyStopper


class ModelTrainer:
    """
    Convenience class to train and validate a model. Call run() to train the model!

    ## Initialization Notes
    1. To specify which GPU to use, set the environment variable. Example: os.environ["CUDA_VISIBLE_DEVICES"]="2"
    2. By default, trains using a SGD optimizer. You can change it using the function [.set_optimizer] before
    calling run()
    3. Similarly, any of the other properties can also be changed before calling run
    4. Turn on reporting when using with Ray Tune

    ## Results
    train_loss, validation_loss
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader_gen: DataLoaderGenerator,
        validation_method: ValidationMethod,
        loss_func: LossFunction,
        early_stopper: Optional[EarlyStopper] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model
        self.loss_func = loss_func
        # Call a reset on the loss tracker
        loss_func.loss_tracker.reset()
        self.dataloader_gen = dataloader_gen
        self.validation_method = validation_method
        self.train_loader: DataLoader
        self.validation_loader: DataLoader
        # Default optimizer
        self.optimizer: Optimizer
        self.optimizer = SGD(self.model.parameters(), lr=3e-4, momentum=0.9)
        self.device = device
        # Trackers
        self.reporting = False
        self.total_epochs = 0
        # Set initial mode to train
        self.mode = "train"
        # Early Stopping
        if early_stopper is None:
            early_stopper = EarlyStopper()
        self.early_stopper = early_stopper
        self.early_stopper.attach_loss_function(loss_func)

    def set_optimizer(self, optimizer_class: Type, kwargs: Dict) -> None:
        """Change the current optimizer. Call this method before calling run to see the effects

        Args:
            optimizer_class (Type): Name of the optimizer class (NOT an Optimizer object. e.g.: SGD, Adam)
            kwargs (Dict): Key Word arguments to be passed into the optimizer
        """
        self.optimizer = optimizer_class(self.model.parameters(), **kwargs)

    def set_batch_size(self, batch_size: int) -> None:
        """Changes DataLoader batch size
        (Because of how PyTorch libraries are defined, changing batchsize requires creating a new DataLoader)
        """
        self.dataloader_gen.change_batch_size(batch_size)
        self.train_loader, self.validation_loader = self.dataloader_gen.generate(self.validation_method)

    def run(self, epochs: int) -> None:
        """Run Training and store results. Each Run resets all old results"""
        self.model = self.model.to(self.device)
        self.train_loader, self.validation_loader = self.dataloader_gen.generate(self.validation_method)

        # Train Model
        for _ in range(epochs):  # loop over the dataset multiple times
            # Training Loop
            self.mode = "train"
            self.model = self.model.train()
            for data in self.train_loader:
                self.single_batch_train_run(data)

            # Validation Loop
            self.mode = "validate"
            self.model = self.model.eval()
            for data in self.validation_loader:
                self.single_batch_validation_run(data)

            # Epcch Update
            self.loss_func.loss_tracker_epoch_update()

            # Reporting
            if self.reporting:
                pass
                # TODO: Implement this part if needed in the future
            self.total_epochs += 1

            # Early Stopping
            if self.early_stopper.check_early_stopping():
                break

    def single_batch_train_run(self, data: Tuple) -> None:
        """Run a single batch of data through the model"""
        inputs = data[DATA_LOADER_INPUT_INDEX]

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(inputs)
        loss = self.loss_func(outputs, data, self.mode)
        loss.backward()
        self.optimizer.step()

    def single_batch_validation_run(self, data: Tuple) -> None:
        """Run a single batch of data through the model for validation purposes"""
        inputs = data[DATA_LOADER_INPUT_INDEX]

        with torch.no_grad():
            outputs = self.model(inputs)
            _ = self.loss_func(outputs, data, self.mode)

    def __str__(self) -> str:
        return f"""
        Model Properties:
        {self.model}
        Data Loader Properties:
        {self.dataloader_gen}
        Validation Method:
        {self.validation_method}
        Loss Function:
        {self.loss_func}
        Optimizer Properties":
        {self.optimizer}
        """
