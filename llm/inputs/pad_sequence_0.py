
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def pad_sequence_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensors
    seq1 = torch.randn(5, 10).numpy()
    seq2 = torch.randn(3, 10).numpy()
    seq3 = torch.randn(7, 10).numpy()
    sequences = [seq1, seq2, seq3]
    input_dict = {
        "sequences": sequences,
        "batch_first": False,
        "padding_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensors, batch_first=True
    seq1 = torch.randint(0, 10, (4, 5)).numpy()
    seq2 = torch.randint(0, 10, (2, 5)).numpy()
    seq3 = torch.randint(0, 10, (6, 5)).numpy()
    sequences = [seq1, seq2, seq3]
    input_dict = {
        "sequences": sequences,
        "batch_first": True,
        "padding_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  Different dimensions
    seq1 = torch.randn(2, 3, 4).numpy()
    seq2 = torch.randn(1, 3, 4).numpy()
    seq3 = torch.randn(3, 3, 4).numpy()
    sequences = [seq1, seq2, seq3]
    input_dict = {
        "sequences": sequences,
        "batch_first": False,
        "padding_value": 1.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.utils.rnn.pad_sequence"] = pad_sequence_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.rnn.pad_sequence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.rnn.pad_sequence'.")

check_valid('torch.nn.utils.rnn.pad_sequence', generated_inputs['torch.nn.utils.rnn.pad_sequence'], lib="torch")
