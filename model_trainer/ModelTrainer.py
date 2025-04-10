"""
A model trainer module used to train and validate a model. This module is designed to be used with PyTorch models and
uses a Composer design pattern to allow for easy customization of the training process.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Type
from torch import nn
import torch
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from model_trainer.core import DATA_LOADER_INPUT_INDEX, LossFunction, ModelMode
from model_trainer.DataLoaderGenerators import DataLoaderGenerator
from model_trainer.validation_methods import ValidationMethod
from model_trainer.early_stopping import EarlyStopper

__all__ = ["ModelTrainer", "ModelTrainerNoisy"]


class ModelTrainer:
    """
    Convenience class to train and validate a model. Call run() to train the model!

    ## Initialization Notes
    1. To specify which GPU to use, set the environment variable. Example: os.environ["CUDA_VISIBLE_DEVICES"]="2"
    2. By default, trains using a SGD optimizer. You can change it using the function [.set_optimizer] before
    calling run()
    3. Similarly, any of the other properties can also be changed before calling run
    4. Turn on reporting when using with Ray Tune
    
    ## Arguments
        model: Fully initialized model to be trained
        dataloader_gen: DataLoaderGenerator object to generate the training and validation data loaders
        validation_method: ValidationMethod object to validate the model
        loss_func: LossFunction object to calculate and track the loss
        early_stopper: EarlyStopper object to stop the training based on some criteria
        verbose: Whether to print the training and validation loss after each epoch
        device: Device to use for training (default: cuda)
        lr_schedulers: List of learning rate schedulers to use (default: None) 
                    For each scheduler that you want to attach, pass a list of dict with the following keys:
                        - scheduler: The scheduler class name
                        - kwargs: The keyword arguments to be passed to the scheduler during initialization
                    exmple: [{"scheduler": torch.optim.lr_scheduler.StepLR, "kwargs": {"step_size": 10, "gamma": 0.1}}]
                    

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
        verbose: bool = False,
        device: torch.device = torch.device("cuda"),
        lr_schedulers: Optional[List[Dict]] = None,
    ):
        # TODO: Implement Learning Rate Schedulers
        self.model = model
        self.model_best_state_dict: Optional[Dict] = None
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
        self.verbose = verbose
        self.total_epochs = 0
        # Set initial mode to train
        self.mode = ModelMode.TRAIN
        # Early Stopping
        if early_stopper is None:
            early_stopper = EarlyStopper()
        self.early_stopper = early_stopper
        self.early_stopper.attach_loss_function(loss_func)
        self.lr_schedulers = []
        if lr_schedulers is not None:
            for scheduler_params in lr_schedulers:
                 self.lr_schedulers.append(scheduler_params["scheduler"](self.optimizer, **scheduler_params["kwargs"]))
        self.early_stopper.reset()
        

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

    def run(self, epochs: int) -> None:
        """Run Training and store results. Each Run resets all old results"""
        self.model = self.model.to(self.device)
        self.train_loader, self.validation_loader = self.dataloader_gen.generate(self.validation_method)
        self.early_stopper.reset()

        # Ensure that both loaders have a non-zero length
        if len(self.train_loader) == 0 or len(self.validation_loader) == 0:
            raise ValueError("Data Loaders have a length of zero. Check the DataLoaders!")

        # Train Model
        for _ in range(epochs):  # loop over the dataset multiple times
            # Training Loop
            self.mode = ModelMode.TRAIN
            self.model = self.model.train()
            for data in self.train_loader:
                self.single_batch_train_run(data)
            
            # Update Learning Rate Schedulers
            if self.lr_schedulers:
                for scheduler in self.lr_schedulers:
                    scheduler.step()

            # Validation Loop
            self.mode = ModelMode.VALIDATE
            self.model = self.model.eval()
            for data in self.validation_loader:
                self.single_batch_validation_run(data)

            # Epcch Update
            self.loss_func.loss_tracker_epoch_update()

            # Reporting
            if self.verbose:
                # Print Losses
                print(
                    f"Epoch: {self.total_epochs}",
                    f"Train Loss: {self.loss_func.loss_tracker.epoch_losses[self.loss_func.train_loss_name][-1]:.4}",
                    f"Validation Loss: {self.loss_func.loss_tracker.epoch_losses[self.loss_func.val_loss_name][-1]:.4}",
                )
            self.total_epochs += 1

            # Early Stopping
            self.early_stopper.check_early_stopping()
            if self.early_stopper.best_loss_updated:
                self.model_best_state_dict = deepcopy(self.model.state_dict())

            if self.early_stopper.time_to_stop:
                self.model.load_state_dict(self.model_best_state_dict)
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


class ModelTrainerNoisy(ModelTrainer):
    """
    A slightly modified version of the ModelTrainer class that adds noise to the input data at each training step
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader_gen: DataLoaderGenerator,
        validation_method: ValidationMethod,
        loss_func: LossFunction,
        noisy_column_indices: List[int],
        noise_std: List[float],
        noise_mean: List[float],
        early_stopper: Optional[EarlyStopper] = None,
        device: torch.device = torch.device("cuda"),
    ):
        """
        A slightly modified version of the ModelTrainer class that adds noise to the input data at each training step

        :param model: The model to be trained
        :param dataloader_gen: The DataLoaderGenerator object that generates the training and validation data loaders
        :param validation_method: The validation method to be used
        :param loss_func: The loss function to be used
        :param noisy_column_indices: The indices of the columns in the input data that should be noisy
        :param noise_std: The standard deviation of the noise to be added to the noisy columns
        :param noise_mean: The mean of the noise to be added to the noisy columns
        :param early_stopper: The EarlyStopper object to be used. If None, the default EarlyStopper is used
        """
        super().__init__(model, dataloader_gen, validation_method, loss_func, early_stopper, device)
        self.noisy_column_indices = noisy_column_indices
        self.noise_std = torch.tensor(noise_std, dtype=torch.float32, device=device).reshape(1, -1)
        self.noise_mean = torch.tensor(noise_mean, dtype=torch.float32, device=device).reshape(1, -1)

    def single_batch_train_run(self, data: Tuple) -> None:
        """Run a single batch of data through the model with added noise"""
        inputs = data[DATA_LOADER_INPUT_INDEX]
        batch_size = inputs.shape[0]
        inputs[:, self.noisy_column_indices] += torch.normal(
            mean=self.noise_mean.expand(batch_size, -1), std=self.noise_std.expand(batch_size, -1)
        )

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(inputs)
        loss = self.loss_func(outputs, data, self.mode)
        loss.backward()
        self.optimizer.step()
