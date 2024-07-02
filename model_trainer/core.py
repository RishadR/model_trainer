"""
Core abstract components for the model trainer library
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, List, Optional
import torch

# CONSTANTS
DATA_LOADER_INPUT_INDEX, DATA_LOADER_LABEL_INDEX, DATA_LOADER_EXTRA_INDEX = 0, 1, 2

class ModelMode(IntEnum):
    """
    Class to represent the mode of the model
    """

    TRAIN = 0
    VALIDATE = 1

    def __str__(self) -> str:
        return self.name.lower()


class LossTracker:
    """
    A class to track the loss values during training. This class is meant to be used with the ModelTrainer class.

    A LossTracker has the following internal attributes:
        1. tracked_losses: A list of the names of the 2 losses to be tracked: name_train_loss, name_val_loss
        2. epoch_losses: A dictionary that tracks the average loss per epoch for each loss
        3. step_loss_sum: A dictionary that tracks the sum of the losses over all the steps within a single epoch
        4. steps_per_epoch_count: A dictionary that tracks the number of steps within a single epoch. Used for averaging
    """

    def __init__(self, name: str):
        self.name = name
        if name == '':
            self.tracked_losses: List[str] = [f"train_loss", f"val_loss"]
        else:
            self.tracked_losses: List[str] = [f"{name}_train_loss", f"{name}_val_loss"]
        self.epoch_losses: Dict[str, List[float]] = {loss_name: [] for loss_name in self.tracked_losses}
        self.step_loss_sum: Dict[str, float] = {loss_name: 0.0 for loss_name in self.tracked_losses}
        self.steps_per_epoch_count: Dict[str, int] = {loss_name: 0 for loss_name in self.tracked_losses}

    def step_update(self, model_mode: ModelMode, loss_value: float) -> None:
        """
        Update the losses for a single step within an epoch by appending the loss to the per_step_losses dictionary
        """
        loss_index = 0 if model_mode == ModelMode.TRAIN else 1
        loss_name = self.tracked_losses[loss_index]
        self.steps_per_epoch_count[loss_name] += 1
        self.step_loss_sum[loss_name] += loss_value

    def epoch_update(self) -> None:
        """
        Update the loss tracker for the current epoch. This is meant to be called at the end of each epoch.

        Averages out the losses from all the steps and places the average onto the epoch_losses list. Clears the
        per step losses for the next epoch
        """
        for loss_name in self.tracked_losses:
            if self.steps_per_epoch_count[loss_name] != 0:  # Avoid division by zero
                self.epoch_losses[loss_name].append(
                    self.step_loss_sum[loss_name] / self.steps_per_epoch_count[loss_name]
                )
                self.step_loss_sum[loss_name] = 0.0  # Reset
                self.steps_per_epoch_count[loss_name] = 0  # Reset

    def reset(self):
        """
        Clears out all saved losses
        """
        for loss_name in self.tracked_losses:
            self.epoch_losses[loss_name] = []
            self.step_loss_sum[loss_name] = 0.0
            self.steps_per_epoch_count[loss_name] = 0


class LossFunction(ABC):
    """
    Base abstract class for all loss functions. All loss functions must inherit from this class and implement the
    following methods
        |1. __call__ (required) : When called, this method should return the loss value
        |2. __str__  (required) : Return a string representation of the loss function
        |3. loss_tracker_epoch_update (optional) : Called at the end of the epoch to perform any necessary operations on
        |                                          the LossTracker object. This should usually include updating the
        |                                          LossTracker object with the average loss for the epoch
        |4. reset (optional) : Reset the loss function
        |5. children (optional) : Return a list of all the children loss functions (if any)
    """

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = ''
        self.name = name
        self.loss_tracker = LossTracker(name)
        self.train_loss_name = self.loss_tracker.tracked_losses[0]
        self.val_loss_name = self.loss_tracker.tracked_losses[1]

    @abstractmethod
    def __call__(self, model_output, dataloader_data, trainer_mode: str) -> torch.Tensor:
        """
        Calculate & return the loss
        :param model_output: The output of the model
        :param dataloader_data: The data from the dataloader (The length of this depends on the DataLoader used)
        :param trainer_mode: The mode of the trainer (train/validate)

        Implementation Notes: When implementing this method, make sure to update the loss tracker using the
        loss_tracker_step_update method
        """

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the loss function
        """

    def loss_tracker_step_update(self, loss_value: float, trainer_mode: ModelMode) -> None:
        """
        The Defualt loss tracker step update method. This is meant to be called at the very end of the __call__ method.
        If you want some custom behavior, you can override this method in the subclass
        """
        self.loss_tracker.step_update(trainer_mode, loss_value)

    def reset(self) -> None:
        """
        Reset
        """
        self.loss_tracker = LossTracker(list(self.loss_tracker.epoch_losses.keys()))

    def loss_tracker_epoch_update(self) -> None:
        """
        Called at the end of the epoch to perform any necessary operations in the ModelTrainer
        """
        self.loss_tracker.epoch_update()

    def children(self) -> List["LossFunction"]:
        """
        Return a list of all the children loss functions (if any)
        """
        return []
