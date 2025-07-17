
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
    num_layers = 2
    nonlinearity = 'tanh'
    bias = True
    batch_first = False
    dropout = 0.0
    bidirectional = False
    dtype = np.float32
    input_np = np.random.randn(5, 3, input_size).astype(dtype)
    hx_np = np.random.randn(num_layers, 3, hidden_size).astype(dtype)
    input = torch.from_numpy(input_np).float()
    hx = torch.from_numpy(hx_np).float()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": torch.float32,
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_size = 5
    hidden_size = 10
    num_layers = 1
    nonlinearity = 'relu'
    bias = False
    batch_first = True
    dropout = 0.5
    bidirectional = True
    dtype = np.float64
    input_np = np.random.randn(2, 4, input_size).astype(dtype)
    hx_np = np.random.randn(2 * num_layers, 4, hidden_size).astype(dtype)
    input = torch.from_numpy(input_np).double()
    hx = torch.from_numpy(hx_np).double()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": torch.float64,
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - unbatched input
    input_size = 7
    hidden_size = 12
    num_layers = 3
    nonlinearity = 'tanh'
    bias = True
    batch_first = False
    dropout = 0.2
    bidirectional = False
    dtype = np.float32
    input_np = np.random.randn(6, input_size).astype(dtype)
    hx_np = np.random.randn(num_layers, hidden_size).astype(dtype)
    input = torch.from_numpy(input_np).float()
    hx = torch.from_numpy(hx_np).float()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": torch.float32,
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - No hx
    input_size = 8
    hidden_size = 15
    num_layers = 1
    nonlinearity = 'relu'
    bias = False
    batch_first = True
    dropout = 0.0
    bidirectional = True
    dtype = np.float64
    input_np = np.random.randn(3, 5, input_size).astype(dtype)
    input = torch.from_numpy(input_np).double()
    hx = None

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": torch.float64,
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - Different shapes, batch_first=False, bidirectional=True
    input_size = 6
    hidden_size = 8
    num_layers = 2
    nonlinearity = 'tanh'
    bias = True
    batch_first = False
    dropout = 0.3
    bidirectional = True
    dtype = np.float32
    input_np = np.random.randn(7, 2, input_size).astype(dtype)
    hx_np = np.random.randn(2 * num_layers, 2, hidden_size).astype(dtype)
    input = torch.from_numpy(input_np).float()
    hx = torch.from_numpy(hx_np).float()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": torch.float32,
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - dropout > 0
    input_size = 9
    hidden_size = 11
    num_layers = 1
    nonlinearity = 'relu'
    bias = False
    batch_first = True
    dropout = 0.7
    bidirectional = False
    dtype = np.float64
    input_np = np.random.randn(4, 6, input_size).astype(dtype)
    hx_np = np.random.randn(num_layers, 6, hidden_size).astype(dtype)
    input = torch.from_numpy(input_np).double()
    hx = torch.from_numpy(hx_np).double()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": torch.float64,
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - batch size 1
    input_size = 4
    hidden_size = 7
    num_layers = 2
    nonlinearity = 'tanh'
    bias = True
    batch_first = False
    dropout = 0.0
    bidirectional = True
    dtype = np.float32
    input_np = np.random.randn(5, 1, input_size).astype(dtype)
    hx_np = np.random.randn(2 * num_layers, 1, hidden_size).astype(dtype)
    input = torch.from_numpy(input_np).float()
    hx = torch.from_numpy(hx_np).float()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": torch.float32,
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - seq_len = 1
    input_size = 12
    hidden_size = 18
    num_layers = 1
    nonlinearity = 'relu'
    bias = False
    batch_first = True
    dropout = 0.1
    bidirectional = False
    dtype = np.float64
    input_np = np.random.randn(2, 1, input_size).astype(dtype)
    hx_np = np.random.randn(num_layers, 2, hidden_size).astype(dtype)
    input = torch.from_numpy(input_np).double()
    hx = torch.from_numpy(hx_np).double()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": torch.float64,
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_size = 3
    hidden_size = 5
    num_layers = 3
    nonlinearity = 'tanh'
    bias = True
    batch_first = False
    dropout = 0.0
    bidirectional = False
    dtype = np.float32
    input_np = np.random.randn(4, 2, input_size).astype(dtype)
    hx_np = np.random.randn(num_layers, 2, hidden_size).astype(dtype)
    input = torch.from_numpy(input_np).float()
    hx = torch.from_numpy(hx_np).float()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": torch.float32,
        "input": input,
        "hx": hx
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_size = 64
    hidden_size = 32
    num_layers = 1
    nonlinearity = 'relu'
    bias = True
    batch_first = True
    dropout = 0.0
    bidirectional = True
    dtype = np.float64
    input_np = np.random.randn(8, 16, input_size).astype(dtype)
    hx_np = np.random.randn(2 * num_layers, 16, hidden_size).astype(dtype)
    input = torch.from_numpy(input_np).double()
    hx = torch.from_numpy(hx_np).double()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "nonlinearity": nonlinearity,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dtype": torch.float64,
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
