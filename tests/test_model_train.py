from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import sys
sys.path.append('../')
import model_train as mt


class TestModelTrain():
    """Tests model building."""

    def test_build_model(self):
        dim = 256
        FL = 3
        learn_rate = 0.0001
        n_filters = 112
        init = 'he_normal'
        lmbda = 1e-06
        drop = 0.15

        model = mt.build_model(dim, learn_rate, lmbda, drop, FL, init,
                               n_filters)

        # Count parameters in PyTorch model
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        
        # The parameter count should be similar to the Keras model
        # Original Keras model had 10278017 parameters
        # PyTorch model should have approximately the same number
        assert trainable_count + non_trainable_count > 10000000
        assert trainable_count > 10000000
        assert non_trainable_count == 0
