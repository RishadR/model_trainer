"""
A decoupled module for visualizing the model training process.
"""

from typing import Dict, List, Literal, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scienceplots  # ignore warning, the plot style requires this import!
from abc import abstractmethod
from rich.table import Table
from rich.console import Console
from model_trainer.core import LossFunction, LossTracker

plt.style.use(["science", "grid"])


class LossVisualizerMixin:
    """
    Class holding the visualization methods for the LossFunction object. Add this as a mixin
    to get the visualization methods for the LossFunction object.
    """

    name: str
    loss_tracker: LossTracker

    @abstractmethod
    def children() -> List[LossFunction]:
        pass

    def plot_losses(
        self,
        plot_type: Literal["joint", "split"] = "joint",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> Figure:
        """
        Plot the losses inside the loss dictionary. This is meant to be used inside the LossTracker Object.
        You have two plotting options:
            1. joint: Plot all the losses in a single plot
            2. split: Plot the train and validation losses in separate plots

        :param losses: Dictionary containing the name of the losses and their values at each epoch
        :param plot_type: Type of plot to be generated, either 'joint' or 'split'
        :param figsize: Size of the figure to be generated. Pass None to autimatically determine the size
        :return: Figure object containing the plot
        """
        ## Determine Figure Size if None
        if figsize is None:
            figsize = (6, 5) if plot_type == "joint" else (10, 5)

        ## Create Figure
        fig, axes = plt.subplots(1, 2 if plot_type == "split" else 1, figsize=figsize, sharey=True)

        losses: Dict[str, List[float]] = {}
        ## Capture the losses
        for loss_func in [self, *self.children()]:
            losses.update(loss_func.loss_tracker.epoch_losses)

        ## Sanity Checks
        if len(losses) == 0:
            print("No losses to plot!")
            return fig

        ## Plotting
        if plot_type == "joint":
            for loss_name, loss_values in losses.items():
                plt.plot(loss_values, label=loss_name)
            plt.ylim(bottom=0.0)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

        elif plot_type == "split":
            train_losses = [loss_name for loss_name in losses.keys() if loss_name.endswith("train_loss")]
            val_losses = [loss_name for loss_name in losses.keys() if loss_name.endswith("val_loss")]
            for loss_name in train_losses:
                axes[0].plot(losses[loss_name], label=loss_name)
            axes[0].set_ylim(bottom=0.0)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            for loss_name in val_losses:
                axes[1].plot(losses[loss_name], label=loss_name)
            axes[1].set_ylim(bottom=0.0)
            axes[1].set_xlabel("Epoch")
            axes[1].legend()

        return fig

    def print_table(self):
        """
        Print the losses in a tabular format using the rich library
        """
        ## Create Table
        table = Table(title="Losses")
        table.add_column(":eyes:", justify="center", style="cyan", no_wrap=True)
        table.add_column("Train Loss", justify="center", style="magenta", no_wrap=True)
        table.add_column("Validation Loss", justify="center", style="green", no_wrap=True)

        ## Add Rows - Self
        loss_func_list = [self, *self.children()]
        for loss_func in loss_func_list:
            train_loss_name = loss_func.loss_tracker.tracked_losses[0]
            val_loss_name = loss_func.loss_tracker.tracked_losses[1]
            if loss_func.loss_tracker.epoch_losses[train_loss_name]:
                train_loss = f'{loss_func.loss_tracker.epoch_losses[train_loss_name][-1]:.4f}'
            else:
                train_loss = ":question_mark:"
            if loss_func.loss_tracker.epoch_losses[val_loss_name]:
                val_loss = f'{loss_func.loss_tracker.epoch_losses[val_loss_name][-1]:.4f}'
            else:
                val_loss = ":question_mark:"
            table.add_row(loss_func.name, train_loss, val_loss)

        ## Print Table
        console = Console()
        console.print(table)
