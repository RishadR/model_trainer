"""
To be deprecated
"""

# import unittest
# import torch
# from torch.nn import MSELoss
# from model_trainer.core import LossTracker
# from model_trainer.loss_funcs import SumLoss, TorchLossWrapper, DynamicWeightLoss


# class TestLostTracker(unittest.TestCase):
#     def setUp(self) -> None:
#         self.loss_tracker = LossTracker(["train_loss", "val_loss"])

#     def test_initiation(self):
#         self.loss_tracker.reset()
#         self.assertEqual(self.loss_tracker.epoch_losses, {"train_loss": [], "val_loss": []})
#         self.assertEqual(self.loss_tracker.step_loss_sum, {"train_loss": [], "val_loss": []})

#     def test_loss_tracker_step_update(self):
#         self.loss_tracker.reset()
#         self.loss_tracker.step_update("train_loss", 1)
#         self.loss_tracker.step_update("train_loss", 2)
#         self.loss_tracker.step_update("val_loss", 3)
#         self.loss_tracker.step_update("val_loss", 4)
#         self.assertEqual(self.loss_tracker.step_loss_sum["train_loss"], [1, 2])
#         self.assertEqual(self.loss_tracker.step_loss_sum["val_loss"], [3, 4])

#     def test_loss_averaging_on_epoch_update(self):
#         self.loss_tracker.reset()
#         self.loss_tracker.step_update("train_loss", 1)
#         self.loss_tracker.step_update("train_loss", 2)
#         self.loss_tracker.step_update("val_loss", 3)
#         self.loss_tracker.step_update("val_loss", 4)
#         self.loss_tracker.epoch_update()
#         self.assertEqual(self.loss_tracker.epoch_losses["train_loss"], [1.5])
#         self.assertEqual(self.loss_tracker.epoch_losses["val_loss"], [3.5])

#     def test_multiple_epoch_updates_work(self):
#         self.loss_tracker.reset()
#         self.loss_tracker.step_update("train_loss", 1)
#         self.loss_tracker.step_update("train_loss", 2)
#         self.loss_tracker.epoch_update()
#         self.loss_tracker.step_update("train_loss", 3)
#         self.loss_tracker.step_update("train_loss", 4)
#         self.loss_tracker.epoch_update()
#         self.assertEqual(self.loss_tracker.epoch_losses["train_loss"], [1.5, 3.5])


# class TestTorchLossWrapper(unittest.TestCase):
#     def setUp(self) -> None:
#         self.torch_loss_object = TorchLossWrapper(MSELoss())

#     def test_call(self):
#         model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
#         dataloader_data = [torch.tensor([1.0, 2.0, 3.0]).cuda()] * 2  # input, labels (ignore the inputs for this one)
#         loss = self.torch_loss_object(model_output, dataloader_data, "train")
#         self.assertEqual(loss.item(), 0.0)  # Model output same as labels

#     def test_tracked_loss_is_float(self):
#         self.torch_loss_object.loss_tracker.reset()
#         model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
#         dataloader_data = [torch.tensor([1.0, 2.0, 3.0]).cuda()] * 2
#         _ = self.torch_loss_object(model_output, dataloader_data, "train")
#         loss = self.torch_loss_object.loss_tracker.step_loss_sum["train_loss"]
#         self.assertIsInstance(loss[0], float)

#     def test_loss_tracker_epoch_ended_correct_length(self):
#         test_length = 10
#         self.torch_loss_object.loss_tracker.reset()
#         model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
#         dataloader_data = [torch.tensor([1.0, 2.0, 3.0]).cuda()] * 2
#         for i in range(10):
#             _ = self.torch_loss_object(model_output, dataloader_data, "train")
#             _ = self.torch_loss_object(model_output, dataloader_data, "train")
#             self.torch_loss_object.loss_tracker_epoch_update()
#         self.assertEqual(len(self.torch_loss_object.loss_tracker.epoch_losses["train_loss"]), test_length)

#     def test_train_validate_modes_write_to_correct_loss(self):
#         self.torch_loss_object.loss_tracker.reset()
#         for __ in range(2):
#             model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
#             dataloader_data = [torch.tensor([1.0, 2.0, 3.0]).cuda()] * 2
#             _ = self.torch_loss_object(model_output, dataloader_data, "train")
#             self.assertEqual(self.torch_loss_object.loss_tracker.step_loss_sum["train_loss"], [0.0])
#             self.assertEqual(self.torch_loss_object.loss_tracker.step_loss_sum["val_loss"], [])
#             _ = self.torch_loss_object(model_output, dataloader_data, "validate")
#             self.assertEqual(self.torch_loss_object.loss_tracker.step_loss_sum["train_loss"], [0.0])
#             self.assertEqual(self.torch_loss_object.loss_tracker.step_loss_sum["val_loss"], [0.0])
#             self.torch_loss_object.loss_tracker_epoch_update()


# class TestTorchLossWithChangingWeight(unittest.TestCase):
#     def setUp(self) -> None:
#         loss_func = TorchLossWrapper(MSELoss())
#         self.loss_nodelay = DynamicWeightLoss(loss_func, 0, 1, 2)
#         self.loss_delay = DynamicWeightLoss(loss_func, 1, 2, 2, 3)

#     def test_weights_applied_correctly(self):
#         self.loss_nodelay.reset()
#         model_output = torch.tensor([1.0]).cuda()
#         dataloader_data = [torch.tensor([2.0]).cuda()] * 2
#         _ = self.loss_nodelay(model_output, dataloader_data, "train")
#         self.loss_nodelay.loss_tracker_epoch_update()
#         _ = self.loss_nodelay(model_output, dataloader_data, "train")
#         self.loss_nodelay.loss_tracker_epoch_update()
#         # First loss = 1, weight = 0, second loss = 1, weight = 1 -> epoch loss = [0.0, 1.0]
#         self.assertEqual(self.loss_nodelay.loss_tracker.epoch_losses["train_loss"], [0.0, 1.0])

#     def test_current_epoch_counter(self):
#         self.loss_nodelay.reset()
#         self.assertEqual(self.loss_nodelay.current_epoch, 0)  # Starts at 1
#         model_output = torch.tensor([1.0]).cuda()
#         dataloader_data = [torch.tensor([2.0]).cuda()] * 2
#         _ = self.loss_nodelay(model_output, dataloader_data, "train")
#         self.loss_nodelay.loss_tracker_epoch_update()
#         self.assertEqual(self.loss_nodelay.current_epoch, 1)  # First epoch
#         _ = self.loss_nodelay(model_output, dataloader_data, "train")
#         self.loss_nodelay.loss_tracker_epoch_update()
#         self.assertEqual(self.loss_nodelay.current_epoch, 1)  # Epoch counter should not increase
#         _ = self.loss_nodelay(model_output, dataloader_data, "train")
#         self.loss_nodelay.loss_tracker_epoch_update()
#         self.assertEqual(self.loss_nodelay.current_epoch, 1)  # Epoch counter should not increase

#     def test_delayed_weight_change(self):
#         self.loss_delay.reset()
#         model_output = torch.tensor([1.0]).cuda()
#         dataloader_data = [torch.tensor([2.0]).cuda()] * 2
#         for i in range(5):
#             _ = self.loss_delay(model_output, dataloader_data, "train")
#             self.loss_delay.loss_tracker_epoch_update()
#         # First three epochs: weight = 1, next two epochs: starts with weight = 1, ends with weight = 2
#         self.assertEqual(self.loss_delay.loss_tracker.epoch_losses["train_loss"], [1.0, 1.0, 1.0, 1.0, 2.0])


# class TestSumLoss(unittest.TestCase):
#     def setUp(self) -> None:
#         self.loss_func1 = TorchLossWrapper(MSELoss(), name="loss1")
#         self.loss_func2 = TorchLossWrapper(MSELoss(), name="loss2")
#         self.loss_func = SumLoss([self.loss_func1, self.loss_func2], [1.0, 1.0])

#     def test_epoch_sum(self):
#         self.loss_func.loss_tracker.reset()
#         model_output = torch.tensor([1.0]).cuda()
#         dataloader_data = [torch.tensor([2.0]).cuda()] * 2
#         _ = self.loss_func(model_output, dataloader_data, "train")
#         self.loss_func.loss_tracker_epoch_update()
#         expected_loss = 2.0
#         self.assertEqual(self.loss_func.loss_tracker.epoch_losses["train_loss"], [expected_loss])

#     def test_weighting(self):
#         weighted_loss = SumLoss([self.loss_func1, self.loss_func2], [1.0, 2.0])
#         weighted_loss.loss_tracker.reset()
#         model_output = torch.tensor([1.0]).cuda()
#         dataloader_data = [torch.tensor([2.0]).cuda()] * 2
#         _ = weighted_loss(model_output, dataloader_data, "train")
#         weighted_loss.loss_tracker_epoch_update()
#         expected_loss = 3.0
#         self.assertEqual(weighted_loss.loss_tracker.epoch_losses["train_loss"], [expected_loss])






# if __name__ == "__main__":
#     unittest.main()
