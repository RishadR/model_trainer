"""
Tests for the new and improved loss functions.
"""

import unittest

import torch
from model_trainer.loss_funcs import SumLoss, TorchLossWrapper
from model_trainer.core import LossTracker, ModelMode


class TestLossTracker(unittest.TestCase):
    def setUp(self) -> None:
        self.loss_tracker = LossTracker("blue_eye_white_dragon")

    def test_initiation(self):
        self.loss_tracker.reset()
        self.assertEqual(len(self.loss_tracker.tracked_losses), 2)
        self.assertTrue(all([x == [] for x in self.loss_tracker.epoch_losses.values()]))
        self.assertTrue(all([x == 0.0 for x in self.loss_tracker.step_loss_sum.values()]))
        self.assertTrue(all([x == 0 for x in self.loss_tracker.steps_per_epoch_count.values()]))

    def test_loss_tracker_step_update(self):
        self.loss_tracker.reset()
        self.loss_tracker.step_update(ModelMode.TRAIN, 1.0)
        self.loss_tracker.step_update(ModelMode.TRAIN, 1.5)
        self.loss_tracker.step_update(ModelMode.VALIDATE, 2.0)
        train_loss_name = self.loss_tracker.tracked_losses[0]
        val_loss_name = self.loss_tracker.tracked_losses[1]
        self.assertEqual(self.loss_tracker.step_loss_sum[train_loss_name], 2.5)
        self.assertEqual(self.loss_tracker.step_loss_sum[val_loss_name], 2.0)
        self.assertEqual(self.loss_tracker.steps_per_epoch_count[train_loss_name], 2)
        self.assertEqual(self.loss_tracker.steps_per_epoch_count[val_loss_name], 1)

    def test_epoch_update(self):
        self.loss_tracker.reset()
        self.loss_tracker.step_update(ModelMode.TRAIN, 1.0)
        self.loss_tracker.step_update(ModelMode.TRAIN, 1.5)
        self.loss_tracker.step_update(ModelMode.VALIDATE, 2.0)
        self.loss_tracker.epoch_update()
        train_loss_name = self.loss_tracker.tracked_losses[0]
        val_loss_name = self.loss_tracker.tracked_losses[1]
        self.assertEqual(len(self.loss_tracker.epoch_losses[train_loss_name]), 1)
        self.assertEqual(len(self.loss_tracker.epoch_losses[val_loss_name]), 1)
        self.assertEqual(self.loss_tracker.epoch_losses[train_loss_name][-1], 1.25)
        self.assertEqual(self.loss_tracker.epoch_losses[val_loss_name][-1], 2.0)


class TestTorchLossWrapper(unittest.TestCase):
    def setUp(self) -> None:
        self.torch_loss_wrapper = TorchLossWrapper(torch.nn.MSELoss(), name="kaiba")

    def test_call(self):
        model_output = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3)
        dataloader_data = [0, torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3)]
        loss = self.torch_loss_wrapper(model_output, dataloader_data, ModelMode.TRAIN)
        self.assertEqual(loss, 0.0)

    def test_step_tracked_properly(self):
        self.torch_loss_wrapper = TorchLossWrapper(torch.nn.MSELoss(), name="kaiba")  # Reset
        model_output = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3)
        dataloader_data = [0, torch.tensor([2.0, 3.0, 4.0]).reshape(1, 3)]
        _ = self.torch_loss_wrapper(model_output, dataloader_data, ModelMode.TRAIN)
        train_loss_name = self.torch_loss_wrapper.loss_tracker.tracked_losses[0]
        self.assertEqual(self.torch_loss_wrapper.loss_tracker.step_loss_sum[train_loss_name], 1.0)
        self.assertEqual(self.torch_loss_wrapper.loss_tracker.steps_per_epoch_count[train_loss_name], 1)

    def test_epoch_tracked_properly(self):
        self.torch_loss_wrapper = TorchLossWrapper(torch.nn.MSELoss(), name="kaiba")  # Reset
        model_output = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3)
        dataloader_data = [0, torch.tensor([2.0, 3.0, 4.0]).reshape(1, 3)]
        _ = self.torch_loss_wrapper(model_output, dataloader_data, ModelMode.TRAIN)
        self.torch_loss_wrapper.loss_tracker_epoch_update()
        train_loss_name = self.torch_loss_wrapper.loss_tracker.tracked_losses[0]
        self.assertEqual(self.torch_loss_wrapper.loss_tracker.epoch_losses[train_loss_name][-1], 1.0)

    def test_multiple_epoch_tracked_properly(self):
        self.torch_loss_wrapper = TorchLossWrapper(torch.nn.MSELoss(), name="kaiba")  # Reset
        model_output = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3)
        dataloader_data = [0, torch.tensor([2.0, 3.0, 4.0]).reshape(1, 3)]
        _ = self.torch_loss_wrapper(model_output, dataloader_data, ModelMode.TRAIN)
        self.torch_loss_wrapper.loss_tracker_epoch_update()
        _ = self.torch_loss_wrapper(model_output, dataloader_data, ModelMode.TRAIN)
        self.torch_loss_wrapper.loss_tracker_epoch_update()
        _ = self.torch_loss_wrapper(model_output, dataloader_data, ModelMode.TRAIN)
        self.torch_loss_wrapper.loss_tracker_epoch_update()
        train_loss_name = self.torch_loss_wrapper.loss_tracker.tracked_losses[0]
        self.assertEqual(len(self.torch_loss_wrapper.loss_tracker.epoch_losses[train_loss_name]), 3)
        self.assertEqual(self.torch_loss_wrapper.loss_tracker.epoch_losses[train_loss_name], [1.0, 1.0, 1.0])

class TestSumLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.loss_func1 = TorchLossWrapper(torch.nn.MSELoss(), name="loss1")
        self.loss_func2 = TorchLossWrapper(torch.nn.MSELoss(), name="loss2")
        self.loss_func = SumLoss([self.loss_func1, self.loss_func2], [1.0, 1.0])
        self.model_output = torch.tensor([1.0]).cuda()
        self.dataloader_data = [torch.tensor([2.0]).cuda()] * 2

    def test_step_loss_sums_correctly(self):
        _ = self.loss_func(self.model_output, self.dataloader_data, ModelMode.TRAIN)
        train_loss_name = self.loss_func.loss_tracker.tracked_losses[0]
        train_loss_name1 = self.loss_func1.loss_tracker.tracked_losses[0]
        train_loss_name2 = self.loss_func2.loss_tracker.tracked_losses[0]
        sum_step_loss = self.loss_func.loss_tracker.step_loss_sum[train_loss_name]
        step_loss1 = self.loss_func1.loss_tracker.step_loss_sum[train_loss_name1]
        step_loss2 = self.loss_func2.loss_tracker.step_loss_sum[train_loss_name2]
        self.assertEqual(sum_step_loss, step_loss1 + step_loss2)

    def test_multiple_step_loss_sums_correctly(self):
        _ = self.loss_func(self.model_output, self.dataloader_data, ModelMode.TRAIN)
        _ = self.loss_func(self.model_output, self.dataloader_data, ModelMode.TRAIN)
        train_loss_name = self.loss_func.loss_tracker.tracked_losses[0]
        train_loss_name1 = self.loss_func1.loss_tracker.tracked_losses[0]
        train_loss_name2 = self.loss_func2.loss_tracker.tracked_losses[0]
        sum_step_loss = self.loss_func.loss_tracker.step_loss_sum[train_loss_name]
        step_loss1 = self.loss_func1.loss_tracker.step_loss_sum[train_loss_name1]
        step_loss2 = self.loss_func2.loss_tracker.step_loss_sum[train_loss_name2]
        self.assertEqual(sum_step_loss, step_loss1 + step_loss2)
    
    def test_epoch_loss_sums_correctly(self):
        _ = self.loss_func(self.model_output, self.dataloader_data, ModelMode.TRAIN)
        self.loss_func.loss_tracker_epoch_update()
        train_loss_name = self.loss_func.loss_tracker.tracked_losses[0]
        train_loss_name1 = self.loss_func1.loss_tracker.tracked_losses[0]
        train_loss_name2 = self.loss_func2.loss_tracker.tracked_losses[0]
        sum_epoch_loss = self.loss_func.loss_tracker.epoch_losses[train_loss_name][-1]
        epoch_loss1 = self.loss_func1.loss_tracker.epoch_losses[train_loss_name1][-1]
        epoch_loss2 = self.loss_func2.loss_tracker.epoch_losses[train_loss_name2][-1]
        self.assertEqual(sum_epoch_loss, epoch_loss1 + epoch_loss2)
    
    def test_weights_work_properly(self):
        loss_func = SumLoss([self.loss_func1, self.loss_func2], [0.5, 0.5])
        _ = loss_func(self.model_output, self.dataloader_data, ModelMode.TRAIN)
        train_loss_name = loss_func.loss_tracker.tracked_losses[0]
        train_loss_name1 = self.loss_func1.loss_tracker.tracked_losses[0]
        train_loss_name2 = self.loss_func2.loss_tracker.tracked_losses[0]
        sum_step_loss = loss_func.loss_tracker.step_loss_sum[train_loss_name]
        step_loss1 = self.loss_func1.loss_tracker.step_loss_sum[train_loss_name1]
        step_loss2 = self.loss_func2.loss_tracker.step_loss_sum[train_loss_name2]
        self.assertEqual(sum_step_loss, 0.5 * step_loss1 + 0.5 * step_loss2)



if __name__ == "__main__":
    unittest.main()
