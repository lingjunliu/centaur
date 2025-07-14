
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def rnncell_inputs():
    list_of_inputs = []

    # Input 1
    input_size = 10
    hidden_size = 20
    bias = True
    nonlinearity = 'tanh'
    dtype = np.float32
    input = np.random.randn(3, input_size).astype(dtype)
    hidden = np.random.randn(3, hidden_size).astype(dtype)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "nonlinearity": nonlinearity,
        "dtype": torch.float32,
        "input": input,
        "hidden": hidden
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_size = 5
    hidden_size = 10
    bias = False
    nonlinearity = 'relu'
    dtype = np.float64
    input = np.random.randn(1, input_size).astype(dtype)
    hidden = np.random.randn(1, hidden_size).astype(dtype)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "nonlinearity": nonlinearity,
        "dtype": torch.float64,
        "input": input,
        "hidden": hidden
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_size = 15
    hidden_size = 5
    bias = True
    nonlinearity = 'tanh'
    dtype = np.float32
    input = np.random.randn(5, input_size).astype(dtype)
    hidden = np.random.randn(5, hidden_size).astype(dtype)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "nonlinearity": nonlinearity,
        "dtype": torch.float32,
        "input": input,
        "hidden": hidden
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_size = 7
    hidden_size = 12
    bias = False
    nonlinearity = 'relu'
    dtype = np.float64
    input = np.random.randn(2, input_size).astype(dtype)
    hidden = np.random.randn(2, hidden_size).astype(dtype)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "nonlinearity": nonlinearity,
        "dtype": torch.float64,
        "input": input,
        "hidden": hidden
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_size = 3
    hidden_size = 8
    bias = True
    nonlinearity = 'tanh'
    dtype = np.float32
    input = np.random.randn(4, input_size).astype(dtype)
    hidden = np.random.randn(4, hidden_size).astype(dtype)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "nonlinearity": nonlinearity,
        "dtype": torch.float32,
        "input": input,
        "hidden": hidden
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_size = 9
    hidden_size = 4
    bias = False
    nonlinearity = 'relu'
    dtype = np.float64
    input = np.random.randn(1, input_size).astype(dtype)
    hidden = np.random.randn(1, hidden_size).astype(dtype)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "nonlinearity": nonlinearity,
        "dtype": torch.float64,
        "input": input,
        "hidden": hidden
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_size = 11
    hidden_size = 13
    bias = True
    nonlinearity = 'tanh'
    dtype = np.float32
    input = np.random.randn(2, input_size).astype(dtype)
    hidden = np.random.randn(2, hidden_size).astype(dtype)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "nonlinearity": nonlinearity,
        "dtype": torch.float32,
        "input": input,
        "hidden": hidden
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_size = 6
    hidden_size = 9
    bias = False
    nonlinearity = 'relu'
    dtype = np.float64
    input = np.random.randn(3, input_size).astype(dtype)
    hidden = np.random.randn(3, hidden_size).astype(dtype)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "nonlinearity": nonlinearity,
        "dtype": torch.float64,
        "input": input,
        "hidden": hidden
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_size = 4
    hidden_size = 7
    bias = True
    nonlinearity = 'tanh'
    dtype = np.float32
    input = np.random.randn(5, input_size).astype(dtype)
    hidden = np.random.randn(5, hidden_size).astype(dtype)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "nonlinearity": nonlinearity,
        "dtype": torch.float32,
        "input": input,
        "hidden": hidden
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_size = 8
    hidden_size = 6
    bias = False
    nonlinearity = 'relu'
    dtype = np.float64
    input = np.random.randn(1, input_size).astype(dtype)
    hidden = np.random.randn(1, hidden_size).astype(dtype)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "nonlinearity": nonlinearity,
        "dtype": torch.float64,
        "input": input,
        "hidden": hidden
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.RNNCell"] = rnncell_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.RNNCell' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.RNNCell'.")

check_valid('torch.nn.RNNCell', generated_inputs['torch.nn.RNNCell'], lib="torch", suffix=0)
