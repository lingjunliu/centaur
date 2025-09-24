
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def max_unpool1d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]).numpy()
    indices_tensor = torch.tensor([[[0, 1, 2, 3]]]).long().numpy()
    kernel_size = (2,)
    stride = (2,)
    padding = 0
    output_size = (1, 1, 8)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices_tensor,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0]]]).numpy()
    indices_tensor = torch.tensor([[[0, 1, 0]]]).long().numpy()
    kernel_size = (3,)
    stride = (1,)
    padding = 1
    output_size = (1, 1, 5)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices_tensor,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - No output size specified
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0]]]).numpy()
    indices_tensor = torch.tensor([[[0, 1, 2]]]).long().numpy()
    kernel_size = (2,)
    stride = (2,)
    padding = 0
    output_size = None

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices_tensor,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxUnpool1d_2"] = max_unpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MaxUnpool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool1d_2'.")

check_valid('torch.nn.MaxUnpool1d', generated_inputs['torch.nn.MaxUnpool1d_2'], lib="torch", suffix=2)
