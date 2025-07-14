
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def GRU_inputs():
    list_of_inputs = []

    # Input 1
    input_size = 10
    hidden_size = 20
    num_layers = 2
    bias = True
    batch_first = False
    dropout = 0.0
    bidirectional = False
    dtype = np.float32
    input = np.random.randn(5, 3, input_size).astype(dtype)
    h_0 = np.random.randn(num_layers, 3, hidden_size).astype(dtype)

    input_dict = {
        "input_size": int(input_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "bias": bool(bias),
        "batch_first": bool(batch_first),
        "dropout": float(dropout),
        "bidirectional": bool(bidirectional),
        "dtype": torch.float32,
        "input": input,
        "h_0": h_0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_size = 5
    hidden_size = 10
    num_layers = 1
    bias = False
    batch_first = True
    dropout = 0.5
    bidirectional = True
    dtype = np.float64
    input = np.random.randn(2, 4, input_size).astype(dtype)
    h_0 = np.random.randn(2 * num_layers, 2, hidden_size).astype(dtype)

    input_dict = {
        "input_size": int(input_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "bias": bool(bias),
        "batch_first": bool(batch_first),
        "dropout": float(dropout),
        "bidirectional": bool(bidirectional),
        "dtype": torch.float64,
        "input": input,
        "h_0": h_0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_size = 7
    hidden_size = 12
    num_layers = 3
    bias = True
    batch_first = False
    dropout = 0.2
    bidirectional = False
    dtype = np.float32
    input = np.random.randn(8, 5, input_size).astype(dtype)
    h_0 = np.random.randn(num_layers, 5, hidden_size).astype(dtype)

    input_dict = {
        "input_size": int(input_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "bias": bool(bias),
        "batch_first": bool(batch_first),
        "dropout": float(dropout),
        "bidirectional": bool(bidirectional),
        "dtype": torch.float32,
        "input": input,
        "h_0": h_0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_size = 3
    hidden_size = 7
    num_layers = 1
    bias = False
    batch_first = True
    dropout = 0.0
    bidirectional = True
    dtype = np.float64
    input = np.random.randn(1, 6, input_size).astype(dtype)
    h_0 = np.random.randn(2 * num_layers, 1, hidden_size).astype(dtype)

    input_dict = {
        "input_size": int(input_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "bias": bool(bias),
        "batch_first": bool(batch_first),
        "dropout": float(dropout),
        "bidirectional": bool(bidirectional),
        "dtype": torch.float64,
        "input": input,
        "h_0": h_0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Unbatched Input
    input_size = 4
    hidden_size = 8
    num_layers = 2
    bias = True
    batch_first = False
    dropout = 0.3
    bidirectional = False
    dtype = np.float32
    input = np.random.randn(7, input_size).astype(dtype)
    h_0 = np.random.randn(num_layers, hidden_size).astype(dtype)

    input_dict = {
        "input_size": int(input_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "bias": bool(bias),
        "batch_first": bool(batch_first),
        "dropout": float(dropout),
        "bidirectional": bool(bidirectional),
        "dtype": torch.float32,
        "input": input,
        "h_0": h_0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_size = 6
    hidden_size = 11
    num_layers = 1
    bias = True
    batch_first = False
    dropout = 0.0
    bidirectional = False
    dtype = np.float32
    input = np.random.randn(4, 2, input_size).astype(dtype)
    h_0 = np.random.randn(num_layers, 2, hidden_size).astype(dtype)

    input_dict = {
        "input_size": int(input_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "bias": bool(bias),
        "batch_first": bool(batch_first),
        "dropout": float(dropout),
        "bidirectional": bool(bidirectional),
        "dtype": torch.float32,
        "input": input,
        "h_0": h_0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_size = 8
    hidden_size = 13
    num_layers = 2
    bias = False
    batch_first = True
    dropout = 0.1
    bidirectional = True
    dtype = np.float64
    input = np.random.randn(3, 5, input_size).astype(dtype)
    h_0 = np.random.randn(2 * num_layers, 3, hidden_size).astype(dtype)

    input_dict = {
        "input_size": int(input_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "bias": bool(bias),
        "batch_first": bool(batch_first),
        "dropout": float(dropout),
        "bidirectional": bool(bidirectional),
        "dtype": torch.float64,
        "input": input,
        "h_0": h_0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.GRU"] = GRU_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.GRU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GRU'.")

check_valid('torch.nn.GRU', generated_inputs['torch.nn.GRU'], lib="torch", suffix=0)
