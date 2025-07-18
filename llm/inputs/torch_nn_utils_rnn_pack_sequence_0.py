
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def pack_sequence_inputs():
    list_of_inputs = []

    # The issue stems from the validation tool's inability to handle a list of numpy arrays
    # or a numpy array of objects, which is the natural representation for sequences of varying lengths.
    # To bypass this, we provide a single stacked numpy array. torch.nn.utils.rnn.pack_sequence
    # can iterate over the first dimension of a tensor, effectively treating it as a list of sequences.
    # This approach implies all sequences in a given input must have the same length.

    # Input 1: Basic case, float32 sequences of the same length
    input_dict_1 = {
        'sequences': np.random.rand(3, 5, 10).astype(np.float32),
        'enforce_sorted': True  # Lengths are equal, so it's sorted
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float64 sequences of the same length
    input_dict_2 = {
        'sequences': np.random.rand(4, 8, 2).astype(np.float64),
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: int32 sequences with negative values
    input_dict_3 = {
        'sequences': np.random.randint(-100, 100, size=(2, 6, 4)).astype(np.int32),
        'enforce_sorted': False # Order doesn't matter as lengths are equal
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: A single sequence in the batch
    input_dict_4 = {
        'sequences': np.random.rand(1, 10, 20).astype(np.float32),
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: int64 sequences, sequence length of 1
    input_dict_5 = {
        'sequences': np.random.randint(-10, 10, size=(5, 1, 3)).astype(np.int64),
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Large number of sequences
    input_dict_6 = {
        'sequences': np.random.randn(10, 5, 5).astype(np.float32),
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Sequences are 1D vectors (represented as 2D array)
    input_dict_7 = {
        'sequences': np.random.rand(8, 10).astype(np.float32),
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Another 1D vector case with float64
    input_dict_8 = {
        'sequences': np.random.rand(4, 20).astype(np.float64),
        'enforce_sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Zero-valued sequences
    input_dict_9 = {
        'sequences': np.zeros((3, 7, 7), dtype=np.float32),
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Sequences with a feature dimension of 1
    input_dict_10 = {
        'sequences': np.random.rand(5, 12, 1).astype(np.float32),
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    return list_of_inputs

generated_inputs["torch.nn.utils.rnn.pack_sequence"] = pack_sequence_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.rnn.pack_sequence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.rnn.pack_sequence'.")

check_valid('torch.nn.utils.rnn.pack_sequence', generated_inputs['torch.nn.utils.rnn.pack_sequence'], lib="torch", suffix=0)
