"""
A decoupled module for visualizing the model training process.
"""

from typing import Dict, List, Literal, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scienceplots  # ignore warning, the plot style requires this import!

plt.style.use(["science", "nature"])


def plot_losses(
    losses: Dict[str, List[float]], plot_type: Literal["joint", "split"], figsize: Optional[Tuple[float, float]] = None
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
        figsize = (8, 6) if plot_type == "joint" else (16, 6)

    ## Create Figure
    fig, axes = plt.subplots(1, 2 if plot_type == "split" else 1, figsize=figsize, sharey=True)

    ## Sanity Checks
    if len(losses) == 0:
        print("No losses to plot!")
        return fig

    ## Plotting
    if plot_type == "joint":
        for loss_name, loss_values in losses.items():
            plt.plot(loss_values, label=loss_name)
            plt.ylim(bottom=0.0)
            plt.legend()

    elif plot_type == "split":
        train_losses = [loss_name for loss_name in losses.keys() if loss_name.endswith("train_loss")]
        val_losses = [loss_name for loss_name in losses.keys() if loss_name.endswith("val_loss")]
        for loss_name in train_losses:
            axes[0].plot(losses[loss_name], label=loss_name)
            axes[0].set_ylim(bottom=0.0)
        axes[0].legend()
        for loss_name in val_losses:
            axes[1].plot(losses[loss_name], label=loss_name)
            axes[1].set_ylim(bottom=0.0)
        axes[1].legend()

    return fig
