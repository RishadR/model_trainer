"""
Visualization test - not included in the test module. Meant to check how the visualization module works.
"""

from torch.nn import MSELoss
import torch
import matplotlib.pyplot as plt
from model_trainer.core import ModelMode
from model_trainer.loss_funcs import TorchLossWrapper, SumLoss

plt.style.use('seaborn-whitegrid')

def test_loss_table():
    loss = TorchLossWrapper(MSELoss(), name="dark_magician")
    model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
    dataloader_data = [torch.tensor([1.0, 3.0, 3.0]).cuda()] * 2  # input, labels (ignore the inputs for this one)
    loss(model_output, dataloader_data, ModelMode.TRAIN)
    loss(model_output, dataloader_data, ModelMode.VALIDATE)
    loss.loss_tracker_epoch_update()
    loss.print_table()


def test_loss_table_sum_loss():
    loss1 = TorchLossWrapper(MSELoss(), name="dark_magician1")
    loss2 = TorchLossWrapper(MSELoss(), name="dark_magician2")
    sum_loss = SumLoss([loss1, loss2], weights=[0.5, 1.0])
    model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
    dataloader_data = [torch.tensor([1.0, 3.0, 3.0]).cuda()] * 2  # input, labels (ignore the inputs for this one)
    sum_loss(model_output, dataloader_data, ModelMode.TRAIN)
    sum_loss(model_output, dataloader_data, ModelMode.VALIDATE)
    sum_loss.loss_tracker_epoch_update()
    sum_loss.print_table()


def test_plots():
    loss1 = TorchLossWrapper(MSELoss(), name="dark_magician1")
    loss2 = TorchLossWrapper(MSELoss(), name="dark_magician2")
    sum_loss = SumLoss([loss1, loss2], weights=[0.5, 1.0])
    model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
    dataloader_data = [torch.tensor([1.0, 3.0, 3.0]).cuda()] * 2  # input, labels (ignore the inputs for this one)
    for _ in range(10):
        sum_loss(model_output, dataloader_data, ModelMode.TRAIN)
        sum_loss(model_output, dataloader_data, ModelMode.VALIDATE)
        sum_loss.loss_tracker_epoch_update()

    sum_loss.plot_losses(figsize=(4, 3))
    plt.show()


if __name__ == "__main__":
    print("Running visualization test...")
    print("Visualization of a Single Loss Table...")
    test_loss_table()
    print("Visualization of a Sum Loss Table...")
    test_loss_table_sum_loss()
    print("Test Plots...")
    test_plots()
    print("Visualization test complete!")
