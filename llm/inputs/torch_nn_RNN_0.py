
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import torch.nn as nn
import numpy as np
import copy

def rnn_inputs():
    list_of_inputs = []

    # Input 1: Basic case (batch_first=False)
    input_size = 10
    hidden_size = 20
    num_layers = 1
    batch_size = 3
    seq_len = 5
    D = 1 
    input_dict = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'nonlinearity': 'tanh',
        'bias': True,
        'batch_first': False,
        'dropout': 0.0,
        'bidirectional': False,
        'dtype': torch.float32,
        'input': torch.randn(seq_len, batch_size, input_size).numpy(),
        'hx': torch.randn(D * num_layers, batch_size, hidden_size).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: batch_first=True
    input_size = 10
    hidden_size = 20
    num_layers = 1
    batch_size = 3
    seq_len = 5
    D = 1 
    input_dict = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'nonlinearity': 'tanh',
        'bias': True,
        'batch_first': True,
        'dropout': 0.0,
        'bidirectional': False,
        'dtype': torch.float32,
        'input': torch.randn(batch_size, seq_len, input_size).numpy(),
        'hx': torch.randn(D * num_layers, batch_size, hidden_size).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-layer RNN
    input_size = 10
    hidden_size = 20
    num_layers = 2
    batch_size = 3
    seq_len = 5
    D = 1
    input_dict = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'nonlinearity': 'tanh',
        'bias': True,
        'batch_first': False,
        'dropout': 0.0,
        'bidirectional': False,
        'dtype': torch.float32,
        'input': torch.randn(seq_len, batch_size, input_size).numpy(),
        'hx': torch.randn(D * num_layers, batch_size, hidden_size).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Bidirectional RNN
    input_size = 10
    hidden_size = 20
    num_layers = 1
    batch_size = 3
    seq_len = 5
    D = 2 
    input_dict = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'nonlinearity': 'tanh',
        'bias': True,
        'batch_first': False,
        'dropout': 0.0,
        'bidirectional': True,
        'dtype': torch.float32,
        'input': torch.randn(seq_len, batch_size, input_size).numpy(),
        'hx': torch.randn(D * num_layers, batch_size, hidden_size).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-layer, Bidirectional, batch_first=True
    input_size = 10
    hidden_size = 20
    num_layers = 2
    batch_size = 3
    seq_len = 5
    D = 2 
    input_dict = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'nonlinearity': 'tanh',
        'bias': True,
        'batch_first': True,
        'dropout': 0.0,
        'bidirectional': True,
        'dtype': torch.float32,
        'input': torch.randn(batch_size, seq_len, input_size).numpy(),
        'hx': torch.randn(D * num_layers, batch_size, hidden_size).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: ReLU non-linearity
    input_size = 10
    hidden_size = 20
    num_layers = 1
    batch_size = 3
    seq_len = 5
    D = 1 
    input_dict = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'nonlinearity': 'relu',
        'bias': True,
        'batch_first': False,
        'dropout': 0.0,
        'bidirectional': False,
        'dtype': torch.float32,
        'input': torch.randn(seq_len, batch_size, input_size).numpy(),
        'hx': torch.randn(D * num_layers, batch_size, hidden_size).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: No bias
    input_size = 10
    hidden_size = 20
    num_layers = 1
    batch_size = 3
    seq_len = 5
    D = 1
    input_dict = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'nonlinearity': 'tanh',
        'bias': False,
        'batch_first': False,
        'dropout': 0.0,
        'bidirectional': False,
        'dtype': torch.float32,
        'input': torch.randn(seq_len, batch_size, input_size).numpy(),
        'hx': torch.randn(D * num_layers, batch_size, hidden_size).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Dropout enabled
    input_size = 10
    hidden_size = 20
    num_layers = 3
    batch_size = 3
    seq_len = 5
    D = 1 
    input_dict = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'nonlinearity': 'tanh',
        'bias': True,
        'batch_first': False,
        'dropout': 0.5,
        'bidirectional': False,
        'dtype': torch.float32,
        'input': torch.randn(seq_len, batch_size, input_size).numpy(),
        'hx': torch.randn(D * num_layers, batch_size, hidden_size).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64 dtype
    input_size = 10
    hidden_size = 20
    num_layers = 1
    batch_size = 3
    seq_len = 5
    D = 1 
    input_dict = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'nonlinearity': 'tanh',
        'bias': True,
        'batch_first': False,
        'dropout': 0.0,
        'bidirectional': False,
        'dtype': torch.float64,
        'input': torch.randn(seq_len, batch_size, input_size, dtype=torch.float64).numpy(),
        'hx': torch.randn(D * num_layers, batch_size, hidden_size, dtype=torch.float64).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different dimensions and unbatched input
    input_size = 5
    hidden_size = 10
    num_layers = 2
    seq_len = 7
    D = 1 
    input_dict = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'nonlinearity': 'tanh',
        'bias': True,
        'batch_first': False,
        'dropout': 0.0,
        'bidirectional': False,
        'dtype': torch.float32,
        'input': torch.randn(seq_len, input_size).numpy(),
        'hx': torch.randn(D * num_layers, hidden_size).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.RNN"] = rnn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.RNN' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.RNN'.")

check_valid('torch.nn.RNN', generated_inputs['torch.nn.RNN'], lib="torch", suffix=0)
