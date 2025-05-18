
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def lstm_cell_inputs():
    list_of_inputs = []

    # Input 1
    input_size = 10
    hidden_size = 20
    input = torch.randn(5, input_size).numpy()
    hx = torch.randn(5, hidden_size).numpy()
    cx = torch.randn(5, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()
    bias = True

    input_dict = {
        "input": input,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_size = 5
    hidden_size = 10
    input = torch.randn(2, input_size).numpy()
    hx = torch.randn(2, hidden_size).numpy()
    cx = torch.randn(2, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()
    bias = True

    input_dict = {
        "input": input,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_size = 7
    hidden_size = 12
    input = torch.randn(3, input_size).numpy()
    hx = torch.randn(3, hidden_size).numpy()
    cx = torch.randn(3, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()
    bias = True

    input_dict = {
        "input": input,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_size = 4
    hidden_size = 8
    input = torch.randn(1, input_size).numpy()
    hx = torch.randn(1, hidden_size).numpy()
    cx = torch.randn(1, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()
    bias = True

    input_dict = {
        "input": input,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_size = 6
    hidden_size = 9
    input = torch.randn(4, input_size).numpy()
    hx = torch.randn(4, hidden_size).numpy()
    cx = torch.randn(4, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()
    bias = True

    input_dict = {
        "input": input,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = lstm_cell_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('LSTMCell', list_of_inputs)
