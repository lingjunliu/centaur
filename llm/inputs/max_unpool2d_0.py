
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def max_unpool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input = torch.randn(1, 1, 2, 2).numpy()
    indices = torch.randint(0, 4, (1, 1, 2, 2)).numpy()
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    output_size = (4, 4)

    input_dict = {
        "input": input,
        "indices": indices,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different kernel size and stride
    input = torch.randn(1, 3, 3, 3).numpy()
    indices = torch.randint(0, 9, (1, 3, 3, 3)).numpy()
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (0, 0)
    output_size = (5, 5)

    input_dict = {
        "input": input,
        "indices": indices,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger input with padding
    input = torch.randn(2, 4, 5, 5).numpy()
    indices = torch.randint(0, 25, (2, 4, 5, 5)).numpy()
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (1, 1)
    output_size = (9, 9)

    input_dict = {
        "input": input,
        "indices": indices,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Odd kernel size and stride
    input = torch.randn(1, 2, 4, 4).numpy()
    indices = torch.randint(0, 9, (1, 2, 4, 4)).numpy()
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    output_size = (7, 7)

    input_dict = {
        "input": input,
        "indices": indices,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Float input
    input = torch.randn(1, 1, 2, 2).float().numpy()
    indices = torch.randint(0, 4, (1, 1, 2, 2)).numpy()
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    output_size = (4, 4)

    input_dict = {
        "input": input,
        "indices": indices,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.max_unpool2d"] = max_unpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.max_unpool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_unpool2d'.")

check_valid('torch.nn.functional.max_unpool2d', generated_inputs['torch.nn.functional.max_unpool2d'], lib="torch")
