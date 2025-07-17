
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def pad_sequence_inputs():
    list_of_inputs = []

    # Input 1: Basic test with different lengths, not batch_first
    sequences = [torch.tensor([1, 2, 3]).numpy(), torch.tensor([4, 5]).numpy(), torch.tensor([6, 7, 8, 9]).numpy()]
    batch_first = False
    padding_value = 0.0
    enforce_sorted = True
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: batch_first = True
    sequences = [torch.tensor([1, 2, 3]).numpy(), torch.tensor([4, 5]).numpy()]
    batch_first = True
    padding_value = -1.0
    enforce_sorted = False
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  enforce_sorted = False, unsorted sequences
    sequences = [torch.tensor([4, 5]).numpy(), torch.tensor([1, 2, 3]).numpy(), torch.tensor([6, 7, 8, 9]).numpy()]
    batch_first = False
    padding_value = 100.0
    enforce_sorted = False
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float tensors
    sequences = [torch.tensor([1.1, 2.2, 3.3]).numpy(), torch.tensor([4.4, 5.5]).numpy()]
    batch_first = False
    padding_value = np.nan
    enforce_sorted = True
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Sequences with negative numbers
    sequences = [torch.tensor([-1, -2, 3]).numpy(), torch.tensor([4, -5]).numpy()]
    batch_first = False
    padding_value = 0.0
    enforce_sorted = True
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Sequences of length 1
    sequences = [torch.tensor([1]).numpy(), torch.tensor([2]).numpy(), torch.tensor([3]).numpy()]
    batch_first = False
    padding_value = 0.0
    enforce_sorted = True
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different padding value
    sequences = [torch.tensor([1, 2, 3]).numpy(), torch.tensor([4, 5]).numpy()]
    batch_first = True
    padding_value = -999.0
    enforce_sorted = True
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single sequence
    sequences = [torch.tensor([1, 2, 3]).numpy()]
    batch_first = False
    padding_value = 0.0
    enforce_sorted = True
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Empty sequence mixed with non-empty sequence
    sequences = [torch.tensor([]).numpy(), torch.tensor([1, 2, 3]).numpy()]
    batch_first = False
    padding_value = 0.0
    enforce_sorted = True
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.utils.rnn.pad_sequence"] = pad_sequence_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.rnn.pad_sequence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.rnn.pad_sequence'.")

check_valid('torch.nn.utils.rnn.pad_sequence', generated_inputs['torch.nn.utils.rnn.pad_sequence'], lib="torch", suffix=0)
