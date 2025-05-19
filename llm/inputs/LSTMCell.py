
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def lstm_cell_inputs():
    list_of_inputs = []
    
    input_size = 10
    hidden_size = 20

    # Input 1: Basic float input
    input1 = torch.randn(5, input_size).numpy()
    hx1 = torch.randn(5, hidden_size).numpy()
    cx1 = torch.randn(5, hidden_size).numpy()
    weight_ih1 = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh1 = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih1 = torch.randn(4 * hidden_size).numpy()
    bias_hh1 = torch.randn(4 * hidden_size).numpy()

    input_dict1 = {
        "input": input1,
        "hx": hx1,
        "cx": cx1,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "weight_ih": weight_ih1,
        "weight_hh": weight_hh1,
        "bias_ih": bias_ih1,
        "bias_hh": bias_hh1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different batch size, no bias
    input2 = torch.randn(1, input_size).numpy()
    hx2 = torch.randn(1, hidden_size).numpy()
    cx2 = torch.randn(1, hidden_size).numpy()
    weight_ih2 = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh2 = torch.randn(4 * hidden_size, hidden_size).numpy()

    input_dict2 = {
        "input": input2,
        "hx": hx2,
        "cx": cx2,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": False,
        "weight_ih": weight_ih2,
        "weight_hh": weight_hh2,
        "bias_ih": None,
        "bias_hh": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs = lstm_cell_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('LSTMCell', generated_inputs)
