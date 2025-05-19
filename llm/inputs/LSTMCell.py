
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
    input_val = torch.randn(5, input_size).numpy()
    hx = torch.randn(5, hidden_size).numpy()
    cx = torch.randn(5, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()

    input_dict = {
        "input": input_val,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different batch size and sizes, no bias
    input_size = 5
    hidden_size = 10
    input_val = torch.randn(3, input_size).numpy()
    hx = torch.randn(3, hidden_size).numpy()
    cx = torch.randn(3, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = np.zeros(4 * hidden_size).astype(np.float32)
    bias_hh = np.zeros(4 * hidden_size).astype(np.float32)

    input_dict = {
        "input": input_val,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": False,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different dtype (double)
    input_size = 12
    hidden_size = 8
    input_val = torch.randn(2, input_size, dtype=torch.float64).numpy()
    hx = torch.randn(2, hidden_size, dtype=torch.float64).numpy()
    cx = torch.randn(2, hidden_size, dtype=torch.float64).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size, dtype=torch.float64).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size, dtype=torch.float64).numpy()
    bias_ih = torch.randn(4 * hidden_size, dtype=torch.float64).numpy()
    bias_hh = torch.randn(4 * hidden_size, dtype=torch.float64).numpy()

    input_dict = {
        "input": input_val,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values, larger input size
    input_size = 30
    hidden_size = 15
    input_val = torch.randn(4, input_size) * -1.0
    input_val = input_val.numpy()
    hx = torch.randn(4, hidden_size) * -1.0
    hx = hx.numpy()
    cx = torch.randn(4, hidden_size) * -1.0
    cx = cx.numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()

    input_dict = {
        "input": input_val,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Bias is False and input is of type Float16
    input_size = 8
    hidden_size = 16
    input_val = torch.randn(1, input_size, dtype=torch.float16).numpy()
    hx = torch.randn(1, hidden_size, dtype=torch.float16).numpy()
    cx = torch.randn(1, hidden_size, dtype=torch.float16).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size, dtype=torch.float16).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size, dtype=torch.float16).numpy()
    bias_ih = np.zeros(4 * hidden_size).astype(np.float16)
    bias_hh = np.zeros(4 * hidden_size).astype(np.float16)

    input_dict = {
        "input": input_val,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": False,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Bias is True and input is of type Float32
    input_size = 8
    hidden_size = 16
    input_val = torch.randn(1, input_size, dtype=torch.float32).numpy()
    hx = torch.randn(1, hidden_size, dtype=torch.float32).numpy()
    cx = torch.randn(1, hidden_size, dtype=torch.float32).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size, dtype=torch.float32).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size, dtype=torch.float32).numpy()
    bias_ih = torch.randn(4 * hidden_size, dtype=torch.float32).numpy()
    bias_hh = torch.randn(4 * hidden_size, dtype=torch.float32).numpy()

    input_dict = {
        "input": input_val,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
