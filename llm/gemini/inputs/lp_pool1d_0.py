
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def lp_pool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float tensor
    input = torch.randn(1, 3, 10).numpy()
    norm_type = 2.0
    kernel_size = 3
    stride = 2
    ceil_mode = False
    input_dict = {
        "input": input,
        "norm_type": norm_type,
        "kernel_size": kernel_size,
        "stride": stride,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different input size and stride
    input = torch.randn(2, 4, 20).numpy()
    norm_type = 1.5
    kernel_size = 5
    stride = 3
    ceil_mode = True
    input_dict = {
        "input": input,
        "norm_type": norm_type,
        "kernel_size": kernel_size,
        "stride": stride,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Integer input
    input = torch.randint(0, 10, (1, 2, 15)).float().numpy() #convert to float
    norm_type = 2.0
    kernel_size = 4
    stride = 1
    ceil_mode = False
    input_dict = {
        "input": input,
        "norm_type": norm_type,
        "kernel_size": kernel_size,
        "stride": stride,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values in input
    input = torch.randn(1, 1, 12) * -1.0
    input = input.numpy()
    norm_type = 3.0
    kernel_size = 2
    stride = 2
    ceil_mode = True
    input_dict = {
        "input": input,
        "norm_type": norm_type,
        "kernel_size": kernel_size,
        "stride": stride,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: norm_type as 1
    input = torch.randn(1, 3, 10).numpy()
    norm_type = 1.0
    kernel_size = 3
    stride = 2
    ceil_mode = False
    input_dict = {
        "input": input,
        "norm_type": norm_type,
        "kernel_size": kernel_size,
        "stride": stride,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Input with only one element.
    input = torch.randn(1, 1, 1).numpy()
    norm_type = 2.0
    kernel_size = 1
    stride = 1
    ceil_mode = False
    input_dict = {
        "input": input,
        "norm_type": norm_type,
        "kernel_size": kernel_size,
        "stride": stride,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.lp_pool1d"] = lp_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.lp_pool1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.lp_pool1d'.")

check_valid('torch.nn.functional.lp_pool1d', generated_inputs['torch.nn.functional.lp_pool1d'], lib="torch")
