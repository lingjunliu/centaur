
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def lstmcell_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input_size = 10
    hidden_size = 20
    bias = True
    input_val = np.random.randn(3, input_size).astype(np.float32)
    h_0_val = np.random.randn(3, hidden_size).astype(np.float32)
    c_0_val = np.random.randn(3, hidden_size).astype(np.float32)
    input_dict = {"input_size": input_size, "hidden_size": hidden_size, "bias": bias, "input": input_val, "h_0": h_0_val, "c_0": c_0_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No bias
    input_size = 5
    hidden_size = 10
    bias = False
    input_val = np.random.randn(2, input_size).astype(np.float32)
    h_0_val = np.random.randn(2, hidden_size).astype(np.float32)
    c_0_val = np.random.randn(2, hidden_size).astype(np.float32)
    input_dict = {"input_size": input_size, "hidden_size": hidden_size, "bias": bias, "input": input_val, "h_0": h_0_val, "c_0": c_0_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different batch size
    input_size = 15
    hidden_size = 25
    bias = True
    input_val = np.random.randn(5, input_size).astype(np.float32)
    h_0_val = np.random.randn(5, hidden_size).astype(np.float32)
    c_0_val = np.random.randn(5, hidden_size).astype(np.float32)
    input_dict = {"input_size": input_size, "hidden_size": hidden_size, "bias": bias, "input": input_val, "h_0": h_0_val, "c_0": c_0_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.LSTMCell"] = lstmcell_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.LSTMCell' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LSTMCell'.")

check_valid('torch.nn.LSTMCell', generated_inputs['torch.nn.LSTMCell'], lib="torch", suffix=0)
