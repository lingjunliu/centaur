
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def rsub_inputs():
    list_of_inputs = []

    # The provided signature {'input': 'tensor', 'other': 'tensor', 'alpha': 'float', 'out': 'tensor'}
    # conflicts with the actual torch.rsub API, which does not accept an 'out' keyword argument.
    # To resolve the TypeError, inputs are generated according to the valid API, omitting the 'out' key.
    # This may reveal a discrepancy in the testing harness if it strictly requires the 'out' key.

    # Input 1: Basic case with 1D float tensors
    input_dict_1 = {
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'other': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'alpha': 2.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D int tensors, positive alpha, broadcasting 'other'
    input_dict_2 = {
        'input': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        'other': np.array([10, 20, 30], dtype=np.int32),
        'alpha': 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Tensors with negative values and fractional alpha
    input_dict_3 = {
        'input': np.array([[-1.5, 2.0], [0.0, -3.5]], dtype=np.float32),
        'other': np.array([[10.0, -5.0], [-2.5, 8.0]], dtype=np.float32),
        'alpha': 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Different dtypes (float64) and negative alpha
    input_dict_4 = {
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'other': np.array([4.0, 5.0, 6.0], dtype=np.float64),
        'alpha': -3.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D tensors and a different alpha
    input_dict_5 = {
        'input': np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        'other': np.ones((2, 2, 2), dtype=np.float32) * 10,
        'alpha': 1.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Scalar 'other' tensor broadcasting to 'input' tensor
    input_dict_6 = {
        'input': np.array([[1, 2], [3, 4]], dtype=np.int64),
        'other': np.array(100, dtype=np.int64),
        'alpha': 10.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Scalar 'input' tensor broadcasting to 'other' tensor
    input_dict_7 = {
        'input': np.array(-5.0, dtype=np.float32),
        'other': np.array([-10.0, 0.0, 10.0], dtype=np.float32),
        'alpha': 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Tensors containing zeros and default alpha
    input_dict_8 = {
        'input': np.zeros((2, 3), dtype=np.float32),
        'other': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        'alpha': 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Broadcasting with a singleton dimension
    input_dict_9 = {
        'input': np.arange(12, dtype=np.float32).reshape(3, 4),
        'other': np.array([[100], [200], [300]], dtype=np.float32),
        'alpha': 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Empty tensors
    input_dict_10 = {
        'input': np.array([], dtype=np.float32),
        'other': np.array([], dtype=np.float32),
        'alpha': 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.rsub"] = rsub_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.rsub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rsub'.")

check_valid('torch.rsub', generated_inputs['torch.rsub'], lib="torch", suffix=0)
