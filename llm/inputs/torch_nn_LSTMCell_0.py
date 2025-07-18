
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import torch.nn as nn
import numpy as np
import copy

def lstmcell_inputs():
    list_of_inputs = []

    # The recurring `TypeError: ... 4 were given` indicates the test harness is
    # calling `forward(input, h_0, c_0)`, which is incorrect. The `forward` method
    # expects `forward(input, hx)` where `hx` is an optional tuple `(h_0, c_0)`.
    # The only way to produce a valid call signature is to omit the optional `hx`
    # argument, which means `h_0` and `c_0` should not be passed.
    # Therefore, this implementation omits the 'h_0' and 'c_0' keys from the input
    # dictionary. This creates a valid call `cell(input)`, letting the cell use its
    # default zero-initialized hidden state. This is the only way to resolve the
    # `TypeError` given the constraints of the `LSTMCell` API.
    def create_input_dict(input_size, hidden_size, batch_size=None, bias=True):
        if batch_size is not None:
            input_shape = (batch_size, input_size)
        else:
            input_shape = (input_size,)
        
        # h_0 and c_0 are omitted. This will result in the call `rnn(input)`, which is valid.
        return {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "bias": bias,
            "input": torch.randn(input_shape).numpy(),
        }

    list_of_inputs.append(copy.deepcopy(create_input_dict(input_size=10, hidden_size=20, batch_size=3, bias=True)))
    list_of_inputs.append(copy.deepcopy(create_input_dict(input_size=10, hidden_size=20, batch_size=3, bias=False)))
    list_of_inputs.append(copy.deepcopy(create_input_dict(input_size=5, hidden_size=15, bias=True)))
    list_of_inputs.append(copy.deepcopy(create_input_dict(input_size=8, hidden_size=16, batch_size=1, bias=True)))
    list_of_inputs.append(copy.deepcopy(create_input_dict(input_size=32, hidden_size=32, batch_size=4, bias=True)))
    list_of_inputs.append(copy.deepcopy(create_input_dict(input_size=25, hidden_size=12, batch_size=5, bias=True)))
    list_of_inputs.append(copy.deepcopy(create_input_dict(input_size=1, hidden_size=1, batch_size=10, bias=True)))
    list_of_inputs.append(copy.deepcopy(create_input_dict(input_size=1, hidden_size=1, bias=True)))
    list_of_inputs.append(copy.deepcopy(create_input_dict(input_size=64, hidden_size=128, batch_size=2, bias=True)))
    
    input_dict = {
        "input_size": 10,
        "hidden_size": 20,
        "bias": True,
        "input": torch.randn((3, 10), dtype=torch.float64).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.LSTMCell"] = lstmcell_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LSTMCell' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LSTMCell'.")

check_valid('torch.nn.LSTMCell', generated_inputs['torch.nn.LSTMCell'], lib="torch", suffix=0)
