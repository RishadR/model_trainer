"""
A decoupled module for visualizing the model training process.
"""

from typing import Dict, List, Literal, Optional, Tuple
from abc import abstractmethod, ABC
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from rich.table import Table
from rich.console import Console
from model_trainer.core import LossFunction, LossTracker


class LossVisualizerMixin(ABC):
    """
    Class holding the visualization methods for the LossFunction object. Add this as a mixin
    to get the visualization methods for the LossFunction object.
    """

    name: str
    loss_tracker: LossTracker

    @abstractmethod
    def children(self) -> List[LossFunction]:
        pass

    def plot_losses(
        self,
        plot_type: Literal["joint", "split"] = "joint",
        figsize: Optional[Tuple[float, float]] = None,
        pin_bottom_to_zero: bool = True,
    ) -> Figure:
        """
        Plot the losses inside the loss dictionary. This is meant to be used inside the LossTracker Object.
        You have two plotting options:
            1. joint: Plot all the losses in a single plot
            2. split: Plot the train and validation losses in separate plots

        :param losses: Dictionary containing the name of the losses and their values at each epoch
        :param plot_type: Type of plot to be generated, either 'joint' or 'split'
        :param figsize: Size of the figure to be generated. Pass None to autimatically determine the size
        :param pin_bottom_to_zero: Whether to pin the bottom of the plot to zero
        :return: Figure object containing the plot
        """
        ## Determine Figure Size if None
        if figsize is None:
            figsize = (6, 5) if plot_type == "joint" else (10, 5)

        ## Get Colormap
        cmap = mpl.cm.get_cmap("viridis")

        ## Create Figure
        fig, axes = plt.subplots(1, 2 if plot_type == "split" else 1, figsize=figsize, sharey=True)

        losses: Dict[str, List[float]] = {}
        ## Capture the losses
        for loss_func in [self, *self.children()]:
            for loss_name, loss_values in loss_func.loss_tracker.epoch_losses.items():
                losses[loss_name] = loss_values

        ## Sanity Checks
        if len(losses) == 0:
            print("No losses to plot!")
            return fig

        ## Plotting
        if plot_type == "joint":
            for index, (loss_name, loss_values) in enumerate(losses.items()):
                plt.plot(loss_values, label=loss_name, color=cmap(index / len(losses)))
            if pin_bottom_to_zero:
                plt.ylim(bottom=0.0)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

        elif plot_type == "split":
            train_losses = list(losses.keys())[::2]
            val_losses = list(losses.keys())[1::2]
            for index, loss_name in enumerate(train_losses):
                axes[0].plot(losses[loss_name], label=loss_name, color=cmap(index / len(train_losses)))
            if pin_bottom_to_zero:
                axes[0].set_ylim(bottom=0.0)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            for index, loss_name in enumerate(val_losses):
                axes[1].plot(losses[loss_name], label=loss_name, color=cmap(index / len(val_losses)))
            if pin_bottom_to_zero:
                axes[1].set_ylim(bottom=0.0)
            axes[1].set_xlabel("Epoch")
            axes[1].legend()

        return fig

    def print_table(self) -> Console:
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
                train_loss = f"{loss_func.loss_tracker.epoch_losses[train_loss_name][-1]:.4f}"
            else:
                train_loss = ":question_mark:"
            if loss_func.loss_tracker.epoch_losses[val_loss_name]:
                val_loss = f"{loss_func.loss_tracker.epoch_losses[val_loss_name][-1]:.4f}"
            else:
                val_loss = ":question_mark:"
            table.add_row(loss_func.name, train_loss, val_loss)

        ## Print Table
        console = Console(record=True)
        console.print(table)

        return console
