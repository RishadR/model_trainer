"""
A set of custom loss function meant to be used with the ModelTrainer Class
"""

from typing import List, Optional, Tuple
from matplotlib.figure import Figure
import torch
from model_trainer.core import LossFunction, LossTracker
from model_trainer.core import DATA_LOADER_LABEL_INDEX
from model_trainer.visualizations import LossVisualizerMixin

__all__ = ["TorchLossWrapper", "SumLoss", "DynamicWeightLoss"]

class TorchLossWrapper(LossFunction, LossVisualizerMixin):
    """
    A simple wrapper around torch.nn loss functions. This lets us seemlessly integrate torch loss functions with our own
    ModelTrainer class.
    """

    def __init__(self, torch_loss_object, column_indices: Optional[List[int]] = None, name: Optional[str] = None):
        """
        :param torch_loss_object: An initialized torch loss object
        :param column_indices: The indices of the columns to be used for the loss calculation. If None, all columns are
        considered. Defaults to None
        """
        super().__init__(name)
        self.loss_func = torch_loss_object
        self.column_indices = column_indices

    def __call__(self, model_output, dataloader_data, trainer_mode):
        if self.column_indices is not None:
            loss = self.loss_func(
                model_output[:, self.column_indices], dataloader_data[DATA_LOADER_LABEL_INDEX][:, self.column_indices]
            )
        else:
            loss = self.loss_func(model_output, dataloader_data[DATA_LOADER_LABEL_INDEX])

        # Update internal loss tracker
        self.loss_tracker_step_update(loss.item(), trainer_mode)
        return loss

    def __str__(self) -> str:
        return f"Torch Loss Function: {self.loss_func}"


class SumLoss(LossFunction, LossVisualizerMixin):
    """
    Sum of multiple loss functions

    Special note: The name of the independent losses tracked inside each LossFunction needs to unique between all losses
    being summed

    Addional notes: The internal loss does not update per step. Rather per epoch. To get per step values, you need to
    look at the loss tracker for the individual loss_funcs/i.e. the constituents
    """

    def __init__(
        self, loss_funcs: List[LossFunction], weights: Optional[List[float]] = None, name: Optional[str] = None
    ):
        if name is None:
            name = "total"
        super().__init__(name)
        ## Sanity checks
        if not self._check_loss_funcs_validity(loss_funcs):
            raise ValueError("Loss names should be unique!")
        if weights is None:
            weights = [1.0 / len(loss_funcs)] * len(loss_funcs)     # Equal weights
        assert len(loss_funcs) == len(weights), "Number of loss functions and weights should match!"

        ## Initialize
        self.loss_funcs = loss_funcs
        self.weights_tensor = torch.tensor(weights, dtype=torch.float32)  # CUDA DEMANDS fp32
        self.weights_list = weights

    def children(self) -> List[LossFunction]:
        return self.loss_funcs

    def __call__(self, model_output, dataloader_data, trainer_mode) -> torch.Tensor:
        # Calculate one loss to get the typing correct
        device = model_output.device if isinstance(model_output, torch.Tensor) else model_output[0].device
        loss = torch.tensor(0.0, device=device, dtype=torch.float32)  # Holds the grads
        for loss_func, weight in zip(self.loss_funcs, self.weights_tensor):
            children_loss_term = weight * loss_func(model_output, dataloader_data, trainer_mode)
            loss = torch.add(loss, children_loss_term)

        # Update internal loss tracker
        self.loss_tracker_step_update(loss.item(), trainer_mode)

        return loss

    def __str__(self) -> str:
        individual_loss_descriptions = [str(loss_func) for loss_func in self.loss_funcs]
        individual_loss_descriptions = "\n".join(individual_loss_descriptions)
        return f"""Sum of multiple loss functions.
        Constituent Losses: {[func.name for func in self.loss_funcs]}
        Weights: {self.weights_list}
        Individual Loss Func Description:
        {individual_loss_descriptions}
        """

    def loss_tracker_epoch_update(self) -> None:
        # Update individual loss trackers
        for loss_func in self.loss_funcs:
            loss_func.loss_tracker_epoch_update()

        # Update self
        self.loss_tracker.epoch_update()

    @staticmethod
    def _check_loss_funcs_validity(loss_funcs: List[LossFunction]) -> bool:
        """
        Check if the loss functions are valid. Returns true if the loss names are unique
        """
        loss_names = [loss_func.name for loss_func in loss_funcs]
        return len(loss_names) == len(set(loss_names))


class DynamicWeightLoss(LossFunction, LossVisualizerMixin):
    """
    Loss whose weight changes linearly as training goes on. This is useful for implementing schedules/annealings
    """

    def __init__(
        self,
        loss_func: LossFunction,
        start_weight: float,
        end_weight: float,
        epoch_count: int,
        start_delay: int = 0,
        name: Optional[str] = None,
    ):
        """
        Loss whose weight changes linearly as training goes on. This is useful for implementing schedules/annealings.
        If start delay is set to 0, the weight will start changing from the first epoch. Otherwise, it will hold the
        start value for the first start_delay epochs before starting to change.

        Args:
            loss_func: The underlying loss function
            start_weight: The weight at the start of the epochs
            end_weight: The weight at the end of the epochs
            epoch_count: The total number of epochs
            start_delay: The number of epochs to wait before starting the weight change. During these epochs, the
            start_weight will be used
            name: The name of the loss function

        Parameters:
            current_epoch: The current epoch number. Current epoch starts at 0 and goes up to
            (total_epochs + start_delay - 1). After that point, it stops changing until reset is called
            total_epochs: The total number of epochs as defined by the user for loss weight scheduling. This does not
            include the start_delay(explained below).
            start_delay: The number of epochs to wait before starting the weight change
            weights: The weights for each epoch
            loss_tracker: The loss tracker object
        """
        super().__init__(name)
        self.loss_func = loss_func
        self.weights = torch.linspace(start_weight, end_weight, epoch_count)
        self.weights = torch.cat([start_weight * torch.ones(start_delay), self.weights])
        self.current_epoch = 0
        self.total_epochs = epoch_count
        self.start_delay = start_delay

    def __call__(self, model_output, dataloader_data, trainer_mode):
        loss = self.weights[self.current_epoch].item() * self.loss_func(model_output, dataloader_data, trainer_mode)
        # Update internal loss tracker
        self.loss_tracker_step_update(loss.item(), trainer_mode)
        return loss

    def loss_tracker_epoch_update(self) -> None:
        if self.current_epoch < self.total_epochs + self.start_delay - 1:
            self.current_epoch += 1
        self.loss_tracker.epoch_update()

    def __str__(self) -> str:
        return f"Loss Function with changing weight: {self.loss_func},\n \
        Start Weight: {self.weights[0].item()}, End Weight: {self.weights[-1].item()}, \
        Total Epochs: {self.total_epochs}, Start Delay: {self.start_delay}"

    def reset(self) -> None:
        self.current_epoch = 0
        self.loss_tracker = LossTracker([self.train_loss_name, self.val_loss_name])
