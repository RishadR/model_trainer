"""
An early stopper class to stop training when the model is not improving.
Inspired by: https://github.com/Bjarten/early-stopping-pytorch

Personal Notes: If time permits, maybe decouple the early stopping rules into separate classes. 
"""

from math import inf
from model_trainer.core import LossFunction


class EarlyStopper:
    """
    Early stopping to stop the training when the loss does not improve after a given patience. These classes can be used
    to compose a ModelTrainer

    Current Rules:
        1. NaN Loss
        2. Validation/Training Loss not improving for a given number of epochs

    :param patience: Number of epochs to wait before stopping the training
    :param delta: Minimum change in the monitored quantity to qualify as an improvement

    Note: If you are implementing a custom ModelTrainer, make sure to call attach_loss_function() during the trainer's
    initialization to attach the loss tracker to the early stopper. Also, call the check_early_stopping() method after
    each epoch to monitor the loss values.
    """

    def __init__(self, patience: int = 20, delta: float = 0.0) -> None:
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss: float = []
        self.current_loss: float = []
        self.loss_tracker = None

    def attach_loss_function(self, loss_function: LossFunction) -> None:
        """
        Attach the loss function to the early stopper to monitor the loss values
        """
        self.loss_tracker = loss_function.loss_tracker
        self.best_loss = [inf] * len(self.loss_tracker.epoch_losses)
        self.current_loss = [inf] * len(self.loss_tracker.epoch_losses)

    def _capture_most_recent_loss(self) -> None:
        """
        Private Function. Capture the most recent loss values from the loss tracker
        """
        for idx, loss_name in enumerate(self.loss_tracker.epoch_losses):
            self.current_loss[idx] = self.loss_tracker.epoch_losses[loss_name][-1]

    def _update_best_loss(self) -> None:
        """
        Private Function. Update the best loss values based on the current loss values
        """
        for idx, loss in enumerate(self.current_loss):
            if loss < self.best_loss[idx]:
                self.best_loss[idx] = loss

    def check_early_stopping(self) -> bool:
        """
        Check if the model should stop training based on the loss values
        """
        ## This ordering is important - DO NOT SWAP!
        self._update_best_loss()    # Best loss takes into account all losses up till **BEFORE** the current epoch
        self._capture_most_recent_loss()    # Current loss is the loss of the current epoch

        ## Check for NaN values
        if any([loss != loss for loss in self.current_loss]):
            return True

        ## Check for Loss Improvement
        if any([loss < best_loss - self.delta for loss, best_loss in zip(self.current_loss, self.best_loss)]):
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
