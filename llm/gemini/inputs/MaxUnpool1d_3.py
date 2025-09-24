
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def MaxUnpool1d_inputs():
    list_of_inputs = []

    # Input 1
    kernel_size = 2
    stride = (2,)
    padding = 0
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]).numpy()
    indices = torch.tensor([[[0, 1, 0, 1]]]).numpy()
    output_size = (1, 1, 8)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices,
        "output_size": tuple(output_size)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    kernel_size = 3
    stride = (1,)
    padding = 1
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0]]]).numpy()
    indices = torch.tensor([[[0, 1, 2]]]).numpy()
    output_size = (1, 1, 5)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices,
        "output_size": tuple(output_size)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    kernel_size = 2
    stride = (1,)
    padding = 0
    input_tensor = torch.tensor([[[1.0, 2.0]]]).numpy()
    indices = torch.tensor([[[0, 1]]]).numpy()
    output_size = (1, 1, 3)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices,
        "output_size": tuple(output_size)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    kernel_size = 3
    stride = (2,)
    padding = 0
    input_tensor = torch.tensor([[[1.0, 2.0]]]).numpy()
    indices = torch.tensor([[[0, 1]]]).numpy()
    output_size = (1, 1, 6)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices,
        "output_size": tuple(output_size)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    kernel_size = 4
    stride = (3,)
    padding = 1
    input_tensor = torch.tensor([[[1.0]]]).numpy()
    indices = torch.tensor([[[0]]]).numpy()
    output_size = (1, 1, 4)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices,
        "output_size": tuple(output_size)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxUnpool1d_3"] = MaxUnpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MaxUnpool1d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool1d_3'.")

check_valid('torch.nn.MaxUnpool1d', generated_inputs['torch.nn.MaxUnpool1d_3'], lib="torch", suffix=3)
