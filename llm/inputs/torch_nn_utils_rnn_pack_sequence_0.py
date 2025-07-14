
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def pack_sequence_inputs():
    list_of_inputs = []

    # Input 1: Basic test
    sequences = [torch.tensor([1, 2, 3]).numpy(), torch.tensor([4, 5]).numpy(), torch.tensor([6]).numpy()]
    enforce_sorted = True
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different lengths
    sequences = [torch.tensor([1, 2, 3, 4]).numpy(), torch.tensor([5, 6, 7]).numpy(), torch.tensor([8, 9]).numpy()]
    enforce_sorted = False
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: unsorted sequences
    sequences = [torch.tensor([8, 9]).numpy(), torch.tensor([1, 2, 3, 4]).numpy(), torch.tensor([5, 6, 7]).numpy()]
    enforce_sorted = False
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: sequences with floats
    sequences = [torch.tensor([1.0, 2.0, 3.0]).numpy(), torch.tensor([4.0, 5.0]).numpy()]
    enforce_sorted = True
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.utils.rnn.pack_sequence"] = pack_sequence_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.rnn.pack_sequence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.rnn.pack_sequence'.")

check_valid('torch.nn.utils.rnn.pack_sequence', generated_inputs['torch.nn.utils.rnn.pack_sequence'], lib="torch", suffix=0)
