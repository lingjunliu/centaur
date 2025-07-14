
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def pad_sequence_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    sequences = [torch.tensor([1, 2, 3]).numpy(), torch.tensor([4, 5]).numpy()]
    batch_first = False
    padding_value = 0.0
    enforce_sorted = True
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: batch_first = True
    sequences = [torch.tensor([1, 2, 3]).numpy(), torch.tensor([4, 5]).numpy()]
    batch_first = True
    padding_value = 0.0
    enforce_sorted = True
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: different padding value
    sequences = [torch.tensor([1, 2, 3]).numpy(), torch.tensor([4, 5]).numpy()]
    batch_first = False
    padding_value = -1.0
    enforce_sorted = True
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: enforce_sorted = False
    sequences = [torch.tensor([1, 2, 3]).numpy(), torch.tensor([4, 5]).numpy()]
    batch_first = False
    padding_value = 0.0
    enforce_sorted = False
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: sequences with different data types
    sequences = [torch.tensor([1.0, 2.0, 3.0]).numpy(), torch.tensor([4.0, 5.0]).numpy()]
    batch_first = False
    padding_value = 0.0
    enforce_sorted = True
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: sequences with different shapes
    sequences = [torch.tensor([[1, 2], [3, 4]]).numpy(), torch.tensor([[5, 6]]).numpy()]
    batch_first = False
    padding_value = 0.0
    enforce_sorted = True
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: sequences with negative values
    sequences = [torch.tensor([-1, -2, -3]).numpy(), torch.tensor([-4, -5]).numpy()]
    batch_first = False
    padding_value = 0.0
    enforce_sorted = True
    input_dict = {"sequences": sequences, "batch_first": batch_first, "padding_value": padding_value, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: sequences with different lengths and different data types
    sequences = [torch.tensor([1, 2, 3, 4, 5]).numpy(), torch.tensor([6.0, 7.0, 8.0]).numpy()]
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
