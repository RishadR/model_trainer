import unittest
import torch
from model_trainer.custom_models import FeatureResidualNetwork, PerceptronBD


class TestPerceptronBD(unittest.TestCase):
    def setUp(self):
        self.in_features = 5
        self.out_features = 1
        self.batch_size = 32
        self.x = torch.rand(self.batch_size, self.in_features)

    def test_correct_output_shape(self):
        model = PerceptronBD([self.in_features, 10, self.out_features])
        model = model.eval()
        y = model(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.out_features))

    def test_model_works_with_dropout(self):
        model = PerceptronBD([self.in_features, 10, self.out_features], [0.5, 0.5])
        model = model.eval()
        _ = model(self.x)

    def test_smallest_model_possible_runs(self):
        model = PerceptronBD([self.in_features, self.out_features])
        model = model.eval()
        _ = model(self.x)

    def test_single_input_single_output_works(self):
        x = torch.rand(self.batch_size, 1)
        model = PerceptronBD([1, 2, 1])
        model = model.eval()
        _ = model(x)


class TestFeatureResidualNetowrk(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.out_features = 5
        self.batch_size = 1
        self.lookup_table = torch.rand(100, self.in_features + self.out_features)
        self.lookup_key_indices = torch.arange(self.in_features, self.out_features + self.in_features)
        self.feature_indices = torch.arange(0, self.in_features)
        self.model = FeatureResidualNetwork(
            [self.in_features, 4, self.lookup_key_indices.shape[0]],
            [0.5, 0.5],
            [self.in_features, 8, self.out_features],
            [0.5, 0.5],
            self.lookup_table,
            self.lookup_key_indices,
            self.feature_indices,
        )

        # Training/Testing with only one row of data messes up BatchNorm's moving-mean calcualtion. Set to eval mode to avoid this
        self.model = self.model.eval()

    def test_network_produces_output(self):
        x = torch.rand(self.batch_size, self.in_features)
        y = self.model(x)
        self.assertEqual(y.shape, (self.batch_size, self.out_features))


if __name__ == "__main__":
    unittest.main()
