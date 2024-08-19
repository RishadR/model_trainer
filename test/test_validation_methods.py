"""
Test the validation methods
"""

import pandas as pd
import unittest
from model_trainer.validation_methods import RandomSplit, CVSplit, HoldOneOut


class TestHoldOneOut(unittest.TestCase):
    def setUp(self) -> None:
        self.held_column = "a"
        self.held_value = 2
        self.table = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4], "b": [9, 2, 7, 1, 2, 1, 3, 3, 4, 6]})
        self.hold_one_out = HoldOneOut(self.held_column, self.held_value)
        self.train_table, self.validation_table = self.hold_one_out.split(self.table)

    def test_split_length(self):
        expected_validation_length = self.table[self.table[self.held_column] == self.held_value].shape[0]
        expected_train_length = self.table.shape[0] - expected_validation_length
        self.assertEqual(self.validation_table.shape[0], expected_validation_length)
        self.assertEqual(self.train_table.shape[0], expected_train_length)

    def test_array_values(self):
        held_values = [2, 3]
        train_table, validation_table = HoldOneOut(self.held_column, held_values).split(self.table)
        expected_validation_length = self.table[self.table[self.held_column].isin(held_values)].shape[0]
        expected_train_length = self.table.shape[0] - expected_validation_length
        self.assertEqual(validation_table.shape[0], expected_validation_length)
        self.assertEqual(train_table.shape[0], expected_train_length)


if __name__ == "__main__":
    unittest.main()
