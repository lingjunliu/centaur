
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def unfold_inputs():
    list_of_inputs = []

    input = torch.randn(2, 5, 3, 4).numpy()
    kernel_size = (2, 3)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3, 10, 12).numpy()
    kernel_size = (4, 5)
    dilation = 2
    padding = 1
    stride = 2
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3, 5, 5).numpy()
    kernel_size = 3
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 4, 7, 8).numpy()
    kernel_size = (2, 2)
    dilation = (2, 1)
    padding = (1, 0)
    stride = (1, 2)

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 2, 6, 6).numpy()
    kernel_size = 5
    dilation = 1
    padding = 2
    stride = 1

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Unfold_1"] = unfold_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Unfold_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Unfold_1'.")

check_valid('torch.nn.Unfold', generated_inputs['torch.nn.Unfold_1'], lib="torch")
