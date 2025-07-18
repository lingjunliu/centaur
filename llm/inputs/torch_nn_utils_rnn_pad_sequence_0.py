
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def pad_sequence_inputs():
    list_of_inputs = []

    # Input 1: Basic case, 2D array, batch_first=True
    sequences = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    input_dict = {
        'sequences': sequences,
        'batch_first': True,
        'padding_value': 0.0,
        'enforce_sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case, 2D array, batch_first=False
    sequences = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
    input_dict = {
        'sequences': sequences,
        'batch_first': False,
        'padding_value': 0.0,
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array (list of 2D sequences), batch_first=True
    sequences = np.random.rand(4, 5, 10).astype(np.float32)
    input_dict = {
        'sequences': sequences,
        'batch_first': True,
        'padding_value': 0.0,
        'enforce_sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, batch_first=False
    sequences = np.random.rand(3, 8, 2).astype(np.float32)
    input_dict = {
        'sequences': sequences,
        'batch_first': False,
        'padding_value': 0.5,
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single sequence
    sequences = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    input_dict = {
        'sequences': sequences,
        'batch_first': True,
        'padding_value': 0.0,
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple sequences of length 1
    sequences = np.array([[10], [20], [30]], dtype=np.float32)
    input_dict = {
        'sequences': sequences,
        'batch_first': False,
        'padding_value': 0.0,
        'enforce_sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values in sequences
    sequences = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float32)
    input_dict = {
        'sequences': sequences,
        'batch_first': True,
        'padding_value': 0.0,
        'enforce_sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64 values
    sequences = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float64)
    input_dict = {
        'sequences': sequences,
        'batch_first': True,
        'padding_value': -99.0,
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty sequences array with a defined shape
    sequences = np.empty((0, 5), dtype=np.float32)
    input_dict = {
        'sequences': sequences,
        'batch_first': True,
        'padding_value': 0.0,
        'enforce_sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Larger number of sequences
    sequences = np.random.rand(10, 5).astype(np.float32)
    input_dict = {
        'sequences': sequences,
        'batch_first': False,
        'padding_value': 1.0,
        'enforce_sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.utils.rnn.pad_sequence"] = pad_sequence_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.rnn.pad_sequence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.rnn.pad_sequence'.")

check_valid('torch.nn.utils.rnn.pad_sequence', generated_inputs['torch.nn.utils.rnn.pad_sequence'], lib="torch", suffix=0)
