
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def rnn_inputs():
    list_of_inputs = []

    # Input 1
    input_size = 10
    hidden_size = 20
    num_layers = 1
    nonlinearity = 'tanh'
    bias = True
    batch_first = False
    dropout = 0.0
    bidirectional = False
    dtype = np.float32
    input_np = np.random.randn(5, 3, input_size).astype(dtype)
    hx_np = np.random.randn(num_layers, 3, hidden_size).astype(dtype)

    input = torch.from_numpy(input_np)
    hx = torch.from_numpy(hx_np)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": 'torch.float32',
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_size = 5
    hidden_size = 10
    num_layers = 2
    nonlinearity = 'relu'
    bias = False
    batch_first = True
    dropout = 0.5
    bidirectional = True
    dtype = np.float64
    input_np = np.random.randn(2, 4, input_size).astype(dtype)
    hx_np = np.random.randn(2 * num_layers, 4, hidden_size).astype(dtype)

    input = torch.from_numpy(input_np)
    hx = torch.from_numpy(hx_np)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": 'torch.float64',
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_size = 15
    hidden_size = 25
    num_layers = 3
    nonlinearity = 'tanh'
    bias = True
    batch_first = False
    dropout = 0.2
    bidirectional = False
    dtype = np.float32
    input_np = np.random.randn(7, 1, input_size).astype(dtype)
    hx_np = np.random.randn(num_layers, 1, hidden_size).astype(dtype)

    input = torch.from_numpy(input_np)
    hx = torch.from_numpy(hx_np)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": 'torch.float32',
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_size = 8
    hidden_size = 12
    num_layers = 1
    nonlinearity = 'relu'
    bias = False
    batch_first = True
    dropout = 0.0
    bidirectional = True
    dtype = np.float64
    input_np = np.random.randn(3, 5, input_size).astype(dtype)
    hx_np = np.random.randn(2 * num_layers, 5, hidden_size).astype(dtype)

    input = torch.from_numpy(input_np)
    hx = torch.from_numpy(hx_np)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": 'torch.float64',
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_size = 20
    hidden_size = 30
    num_layers = 2
    nonlinearity = 'tanh'
    bias = True
    batch_first = False
    dropout = 0.8
    bidirectional = False
    dtype = np.float32
    input_np = np.random.randn(4, 2, input_size).astype(dtype)
    hx_np = np.random.randn(num_layers, 2, hidden_size).astype(dtype)

    input = torch.from_numpy(input_np)
    hx = torch.from_numpy(hx_np)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": 'torch.float32',
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_size = 3
    hidden_size = 7
    num_layers = 4
    nonlinearity = 'relu'
    bias = False
    batch_first = True
    dropout = 0.3
    bidirectional = True
    dtype = np.float64
    input_np = np.random.randn(5, 6, input_size).astype(dtype)
    hx_np = np.random.randn(2 * num_layers, 6, hidden_size).astype(dtype)

    input = torch.from_numpy(input_np)
    hx = torch.from_numpy(hx_np)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": 'torch.float64',
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7 - unbatched input
    input_size = 10
    hidden_size = 20
    num_layers = 1
    nonlinearity = 'tanh'
    bias = True
    batch_first = False
    dropout = 0.0
    bidirectional = False
    dtype = np.float32
    input_np = np.random.randn(5, input_size).astype(dtype)
    hx_np = np.random.randn(num_layers, hidden_size).astype(dtype)

    input = torch.from_numpy(input_np)
    hx = torch.from_numpy(hx_np)


    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": 'torch.float32',
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - different dtype
    input_size = 5
    hidden_size = 10
    num_layers = 2
    nonlinearity = 'relu'
    bias = False
    batch_first = True
    dropout = 0.5
    bidirectional = True
    dtype = np.float16
    input_np = np.random.randn(2, 4, input_size).astype(dtype)
    hx_np = np.random.randn(2 * num_layers, 4, hidden_size).astype(dtype)
    
    input = torch.from_numpy(input_np)
    hx = torch.from_numpy(hx_np)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": 'torch.float16',
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9 - Zero values
    input_size = 10
    hidden_size = 20
    num_layers = 1
    nonlinearity = 'tanh'
    bias = True
    batch_first = False
    dropout = 0.0
    bidirectional = False
    dtype = np.float32
    input_np = np.zeros((5, 3, input_size)).astype(dtype)
    hx_np = np.zeros((num_layers, 3, hidden_size)).astype(dtype)

    input = torch.from_numpy(input_np)
    hx = torch.from_numpy(hx_np)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": 'torch.float32',
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - Small values
    input_size = 5
    hidden_size = 10
    num_layers = 2
    nonlinearity = 'relu'
    bias = False
    batch_first = True
    dropout = 0.5
    bidirectional = True
    dtype = np.float64
    input_np = np.random.rand(2, 4, input_size).astype(dtype) * 0.001
    hx_np = np.random.rand(2 * num_layers, 4, hidden_size).astype(dtype)

    input = torch.from_numpy(input_np)
    hx = torch.from_numpy(hx_np)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": 'torch.float64',
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.RNN"] = rnn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.RNN' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.RNN'.")

check_valid('torch.nn.RNN', generated_inputs['torch.nn.RNN'], lib="torch", suffix=0)
