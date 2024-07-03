"""
Test the early stopping module.
"""

import unittest
from torch.nn import MSELoss
from model_trainer.early_stopping import EarlyStopper
from model_trainer.loss_funcs import TorchLossWrapper


class TestEarlyStopper(unittest.TestCase):
    def setUp(self):
        self.loss_func = TorchLossWrapper(MSELoss())
        self.loss_func.loss_tracker.epoch_losses = {"train_loss": []}

    def test_capture_most_recent_loss(self):
        early_stopper = EarlyStopper(patience=2, delta=0.0)
        early_stopper.attach_loss_function(self.loss_func)
        self.loss_func.loss_tracker.epoch_losses = {"train_loss": [0.0]}
        early_stopper._capture_most_recent_loss()  # pylint: disable=protected-access
        self.assertEqual(early_stopper.current_loss, [0.0])

    def test_update_best_loss(self):
        early_stopper = EarlyStopper(patience=2, delta=0.0)
        early_stopper.attach_loss_function(self.loss_func)
        self.loss_func.loss_tracker.epoch_losses = {"train_loss": [2.0]}
        early_stopper.check_early_stopping()
        self.loss_func.loss_tracker.epoch_losses = {"train_loss": [2.0, 1.0]}
        early_stopper.check_early_stopping()
        self.loss_func.loss_tracker.epoch_losses = {"train_loss": [2.0, 1.0, 1.5]}
        early_stopper.check_early_stopping()
        self.assertEqual(early_stopper.best_loss, [1.0])

    def test_stops_on_nan(self):
        early_stopper = EarlyStopper(patience=2, delta=0.0)
        early_stopper.attach_loss_function(self.loss_func)
        self.loss_func.loss_tracker.epoch_losses = {"train_loss": [float("nan")]}
        self.assertTrue(early_stopper.check_early_stopping())

    def test_stops_on_no_improvement(self):
        early_stopper = EarlyStopper(patience=2, delta=0.0)
        early_stopper.attach_loss_function(self.loss_func)
        self.loss_func.loss_tracker.epoch_losses = {"train_loss": [2.0]}
        early_stopper.check_early_stopping()
        self.loss_func.loss_tracker.epoch_losses = {"train_loss": [2.0, 2.0]}
        early_stopper.check_early_stopping()
        self.loss_func.loss_tracker.epoch_losses = {"train_loss": [2.0, 2.0, 2.0]}
        self.assertTrue(early_stopper.check_early_stopping())
    
    def test_resets_when_error_improves(self):
        early_stopper = EarlyStopper(patience=2, delta=0.0)
        early_stopper.attach_loss_function(self.loss_func)
        self.loss_func.loss_tracker.epoch_losses = {"train_loss": [2.0]}
        early_stopper.check_early_stopping()
        # Error improves -> counter resets
        self.loss_func.loss_tracker.epoch_losses = {"train_loss": [2.0, 1.0]}
        early_stopper.check_early_stopping()
        self.assertTrue(early_stopper.counter == 0)
        # Error improves -> counter resets
        self.loss_func.loss_tracker.epoch_losses = {"train_loss": [2.0, 1.0, 0.7]}
        early_stopper.check_early_stopping()
        self.assertTrue(early_stopper.counter == 0)
        # Error worsens -> counter increments
        self.loss_func.loss_tracker.epoch_losses = {"train_loss": [2.0, 1.0, 0.7, 0.8]}
        early_stopper.check_early_stopping()
        self.assertTrue(early_stopper.counter == 1)
        


if __name__ == "__main__":
    unittest.main()
