"""
An early stopper class to stop training when the model is not improving.
Inspired by: https://github.com/Bjarten/early-stopping-pytorch

Personal Notes: If time permits, maybe decouple the early stopping rules into separate classes. 
"""

from abc import ABC, abstractmethod
from typing import Optional
from math import inf
from model_trainer.core import LossFunction, LossTracker

__all__ = ["EarlyStopper", "EarlyStopperCondition"]


class EarlyStopper:
    """
    Early stopping to stop the training when the validation loss does not improve after a given patience. These classes
    can be used to compose a ModelTrainer

    Current Rules:
    --------------
        1. NaN Loss
        2. Validation/Training Loss not improving for a given number of epochs

    Args:
    -----
    :param patience: Number of epochs to wait before stopping the training
    :param delta_for_patience: Minimum change in the monitored quantity to qualify as an improvement

    Custom Usage Instructions:
    --------------------------
    1. Attach the loss function to the early stopper using the attach_loss_function method. The val_loss_name of this
    loss function will be used to monitor the loss values.
    2. Call reset() before starting a new training session - this clears out the previous training session's best
    loss values, patience counter, etc.
    3. Call check_early_stopping() after each epoch to check if the model should stop training
    4. Check best_loss_updated property to see if the best loss value was updated in the last epoch
    5. Check time_to_stop property to see if the model should stop training
    6. Alternatively, check_early_stopping() produces a boolen analogous to time_to_stop


    """

    def __init__(self, patience: int = 10, delta_for_patience: float = 0.0) -> None:
        self.patience = patience
        self.delta = delta_for_patience
        self.patience_counter = 0
        self.best_val_loss: float = inf
        self.current_val_loss: float = inf
        self.loss_tracker: Optional[LossTracker] = None
        self.best_loss_updated: bool = False
        self.best_loss_epoch: int = 0
        self.time_to_stop: bool = False
        self.train_loss_name: str = ""
        self.val_loss_name: str = ""

    def attach_loss_function(self, loss_function: LossFunction) -> None:
        """
        Attach the loss function to the early stopper to monitor the loss values
        """
        self.loss_tracker = loss_function.loss_tracker
        self.val_loss_name = loss_function.val_loss_name
        self.train_loss_name = loss_function.train_loss_name

    def check_early_stopping(self) -> bool:
        """
        Check if the model should stop training based on the loss values
        """
        self.best_loss_updated: bool = False
        if self.loss_tracker is None:
            raise ValueError("Loss Tracker not attached to the Early Stopper. Please attach the loss tracker first.")

        if self.time_to_stop:
            return True

        if len(self.loss_tracker.epoch_losses[self.val_loss_name]) == 0:
            return False

        ## Update Loss Values
        self.current_val_loss = self.loss_tracker.epoch_losses[self.val_loss_name][-1]
        current_train_loss = self.loss_tracker.epoch_losses[self.train_loss_name][-1]

        ## Check for NaN values
        if (self.current_val_loss != self.current_val_loss) or (current_train_loss != current_train_loss):
            self.time_to_stop = True
            return True

        ## Check Train Loss and Validation Loss for stopping condition & Best Loss
        if (self.current_val_loss >= current_train_loss) and (self.current_val_loss > self.best_val_loss + self.delta):
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.time_to_stop = True
                return True
        else:
            self.best_val_loss = self.current_val_loss
            self.best_loss_epoch = len(self.loss_tracker.epoch_losses[self.val_loss_name])
            self.best_loss_updated: bool = True
            self.patience_counter = 0
        ## Note: Resist the urge to return self.time_to_stop and replace all the intermediate returns steps. The inter
        ## -mediate steps break out of the flow and prevents long-winded if-else statements
        return False

    def reset(self) -> None:
        """
        Reset the early stopper to its initial state. Call this method before starting a new training session
        """
        self.patience_counter = 0
        self.best_val_loss = inf
        self.current_val_loss = inf
        self.best_loss_updated = False
        self.best_loss_epoch = 0
        self.time_to_stop = False


class EarlyStopperCondition(ABC):
    """
    Abstract class for early stopping conditions
    """

    def __init__(self, stopper: EarlyStopper):
        self.stopper = stopper

    @abstractmethod
    def check_early_stopping(self) -> bool:
        pass


# TODO: Move all early stopping conditions as separate classes and inherit from EarlyStopperCondition
