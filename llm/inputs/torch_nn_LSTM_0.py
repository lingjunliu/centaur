
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def lstm_inputs():
    list_of_inputs = []

    # Input 1
    input_size = 10
    hidden_size = 20
    num_layers = 1
    bias = True
    batch_first = False
    dropout = 0.0
    bidirectional = False
    proj_size = 0
    dtype = torch.float32
    input_tensor = torch.randn(5, 3, input_size)
    h_0_tensor = torch.randn(num_layers, 3, hidden_size)
    c_0_tensor = torch.randn(num_layers, 3, hidden_size)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "proj_size": proj_size,
        "dtype": dtype,
        "input": input_tensor.numpy(),
        "h_0": h_0_tensor.numpy(),
        "c_0": c_0_tensor.numpy()
    }

    inner_dict = {
        "args": [input_tensor, (h_0_tensor, c_0_tensor)],
        "kwargs": {}
    }

    input_dict["inner"] = inner_dict
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_size = 5
    hidden_size = 10
    num_layers = 2
    bias = False
    batch_first = True
    dropout = 0.5
    bidirectional = True
    proj_size = 0
    dtype = torch.float64
    input_tensor = torch.randn(2, 4, input_size)
    h_0_tensor = torch.randn(2 * num_layers, 4, hidden_size)
    c_0_tensor = torch.randn(2 * num_layers, 4, hidden_size)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "proj_size": proj_size,
        "dtype": dtype,
        "input": input_tensor.numpy(),
        "h_0": h_0_tensor.numpy(),
        "c_0": c_0_tensor.numpy()
    }
    inner_dict = {
        "args": [input_tensor, (h_0_tensor, c_0_tensor)],
        "kwargs": {}
    }
    input_dict["inner"] = inner_dict
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_size = 7
    hidden_size = 15
    num_layers = 3
    bias = True
    batch_first = False
    dropout = 0.2
    bidirectional = False
    proj_size = 8
    dtype = torch.float32
    input_tensor = torch.randn(10, 1, input_size)
    h_0_tensor = torch.randn(num_layers, 1, proj_size)
    c_0_tensor = torch.randn(num_layers, 1, hidden_size)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "proj_size": proj_size,
        "dtype": dtype,
        "input": input_tensor.numpy(),
        "h_0": h_0_tensor.numpy(),
        "c_0": c_0_tensor.numpy()
    }
    inner_dict = {
        "args": [input_tensor, (h_0_tensor, c_0_tensor)],
        "kwargs": {}
    }
    input_dict["inner"] = inner_dict
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_size = 3
    hidden_size = 6
    num_layers = 1
    bias = False
    batch_first = True
    dropout = 0.0
    bidirectional = True
    proj_size = 0
    dtype = torch.float64
    input_tensor = torch.randn(1, 5, input_size)
    h_0_tensor = torch.randn(2 * num_layers, 5, hidden_size)
    c_0_tensor = torch.randn(2 * num_layers, 5, hidden_size)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "proj_size": proj_size,
        "dtype": dtype,
        "input": input_tensor.numpy(),
        "h_0": h_0_tensor.numpy(),
        "c_0": c_0_tensor.numpy()
    }
    inner_dict = {
        "args": [input_tensor, (h_0_tensor, c_0_tensor)],
        "kwargs": {}
    }
    input_dict["inner"] = inner_dict
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_size = 12
    hidden_size = 8
    num_layers = 2
    bias = True
    batch_first = False
    dropout = 0.8
    bidirectional = False
    proj_size = 0
    dtype = torch.float32
    input_tensor = torch.randn(3, 2, input_size)
    h_0_tensor = torch.randn(num_layers, 2, hidden_size)
    c_0_tensor = torch.randn(num_layers, 2, hidden_size)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "proj_size": proj_size,
        "dtype": dtype,
        "input": input_tensor.numpy(),
        "h_0": h_0_tensor.numpy(),
        "c_0": c_0_tensor.numpy()
    }
    inner_dict = {
        "args": [input_tensor, (h_0_tensor, c_0_tensor)],
        "kwargs": {}
    }
    input_dict["inner"] = inner_dict
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_size = 4
    hidden_size = 7
    num_layers = 1
    bias = False
    batch_first = True
    dropout = 0.1
    bidirectional = True
    proj_size = 3
    dtype = torch.float64
    input_tensor = torch.randn(5, 4, input_size)
    h_0_tensor = torch.randn(2 * num_layers, 4, proj_size)
    c_0_tensor = torch.randn(2 * num_layers, 4, hidden_size)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "proj_size": proj_size,
        "dtype": dtype,
        "input": input_tensor.numpy(),
        "h_0": h_0_tensor.numpy(),
        "c_0": c_0_tensor.numpy()
    }
    inner_dict = {
        "args": [input_tensor, (h_0_tensor, c_0_tensor)],
        "kwargs": {}
    }
    input_dict["inner"] = inner_dict
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_size = 9
    hidden_size = 11
    num_layers = 3
    bias = True
    batch_first = False
    dropout = 0.3
    bidirectional = False
    proj_size = 0
    dtype = torch.float32
    input_tensor = torch.randn(6, 3, input_size)
    h_0_tensor = torch.randn(num_layers, 3, hidden_size)
    c_0_tensor = torch.randn(num_layers, 3, hidden_size)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "proj_size": proj_size,
        "dtype": dtype,
        "input": input_tensor.numpy(),
        "h_0": h_0_tensor.numpy(),
        "c_0": c_0_tensor.numpy()
    }
    inner_dict = {
        "args": [input_tensor, (h_0_tensor, c_0_tensor)],
        "kwargs": {}
    }
    input_dict["inner"] = inner_dict
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_size = 6
    hidden_size = 9
    num_layers = 2
    bias = False
    batch_first = True
    dropout = 0.6
    bidirectional = True
    proj_size = 4
    dtype = torch.float64
    input_tensor = torch.randn(7, 2, input_size)
    h_0_tensor = torch.randn(2 * num_layers, 2, proj_size)
    c_0_tensor = torch.randn(2 * num_layers, 2, hidden_size)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "proj_size": proj_size,
        "dtype": dtype,
        "input": input_tensor.numpy(),
        "h_0": h_0_tensor.numpy(),
        "c_0": c_0_tensor.numpy()
    }
    inner_dict = {
        "args": [input_tensor, (h_0_tensor, c_0_tensor)],
        "kwargs": {}
    }
    input_dict["inner"] = inner_dict
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    input_size = 11
    hidden_size = 13
    num_layers = 4
    bias = True
    batch_first = False
    dropout = 0.4
    bidirectional = False
    proj_size = 0
    dtype = torch.float32
    input_tensor = torch.randn(8, 5, input_size)
    h_0_tensor = torch.randn(num_layers, 5, hidden_size)
    c_0_tensor = torch.randn(num_layers, 5, hidden_size)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "proj_size": proj_size,
        "dtype": dtype,
        "input": input_tensor.numpy(),
        "h_0": h_0_tensor.numpy(),
        "c_0": c_0_tensor.numpy()
    }
    inner_dict = {
        "args": [input_tensor, (h_0_tensor, c_0_tensor)],
        "kwargs": {}
    }
    input_dict["inner"] = inner_dict
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_size = 8
    hidden_size = 12
    num_layers = 3
    bias = False
    batch_first = True
    dropout = 0.7
    bidirectional = True
    proj_size = 5
    dtype = torch.float64
    input_tensor = torch.randn(9, 3, input_size)
    h_0_tensor = torch.randn(2 * num_layers, 3, proj_size)
    c_0_tensor = torch.randn(2 * num_layers, 3, hidden_size)

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "proj_size": proj_size,
        "dtype": dtype,
        "input": input_tensor.numpy(),
        "h_0": h_0_tensor.numpy(),
        "c_0": c_0_tensor.numpy()
    }
    inner_dict = {
        "args": [input_tensor, (h_0_tensor, c_0_tensor)],
        "kwargs": {}
    }
    input_dict["inner"] = inner_dict
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.LSTM"] = lstm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.LSTM' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LSTM'.")

check_valid('torch.nn.LSTM', generated_inputs['torch.nn.LSTM'], lib="torch", suffix=0)
