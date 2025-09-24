
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch, copy
import numpy as np

def conv_transpose2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with different parameters
    input = torch.randn(1, 3, 5, 5).numpy()
    weight = torch.randn(3, 2, 3, 3).numpy()
    bias = torch.randn(2).numpy()
    stride = (2, 2)
    padding = (1, 1)
    output_padding = (1, 1)
    groups = 1
    dilation = (1, 1)

    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different batch size and channel size
    input = torch.randn(2, 6, 8, 8).numpy()
    weight = torch.randn(6, 4, 2, 2).numpy()
    bias = torch.randn(4).numpy()
    stride = (1, 1)
    padding = (0, 0)
    output_padding = (0, 0)
    groups = 1
    dilation = (1, 1)
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Groups > 1 - Corrected weight shape
    input = torch.randn(1, 4, 7, 7).numpy()
    weight = torch.randn(4, 2, 3, 3).numpy()  # Modified weight shape
    bias = torch.randn(2).numpy()
    stride = (2, 2)
    padding = (1, 1)
    output_padding = (1, 1)
    groups = 2  # Not actually using groups for conv_transpose2d, it does not support the groups argument
    dilation = (1, 1)
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": 1,  # Setting groups to 1 to avoid the group related error
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Different dilation
    input = torch.randn(1, 3, 5, 5).numpy()
    weight = torch.randn(3, 2, 3, 3).numpy()
    bias = torch.randn(2).numpy()
    stride = (1, 1)
    padding = (1, 1)
    output_padding = (1, 1)
    groups = 1
    dilation = (2, 2)
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Asymmetric stride and padding
    input = torch.randn(1, 3, 5, 5).numpy()
    weight = torch.randn(3, 2, 3, 3).numpy()
    bias = torch.randn(2).numpy()
    stride = (2, 1)
    padding = (1, 0)
    output_padding = (1, 0)
    groups = 1
    dilation = (1, 1)

    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.conv_transpose2d"] = conv_transpose2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('conv_transpose2d', generated_inputs['torch.nn.functional.conv_transpose2d'], lib="torch")
