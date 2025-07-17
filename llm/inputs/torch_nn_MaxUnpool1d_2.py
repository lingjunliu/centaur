
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def max_unpool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input_val = torch.tensor([[[1.0, 2.0, 3.0]]]).numpy()
    indices_val = torch.tensor([[[1, 3, 5]]]).numpy()
    kernel_size_val = (2,)
    stride_val = (2,)
    padding_val = 0
    output_size_val = (1, 1, 7)

    input_dict = {
        "kernel_size": kernel_size_val,
        "stride": stride_val,
        "padding": padding_val,
        "input": input_val,
        "indices": indices_val,
        "output_size": output_size_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxUnpool1d_2"] = max_unpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MaxUnpool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool1d_2'.")

check_valid('torch.nn.MaxUnpool1d', generated_inputs['torch.nn.MaxUnpool1d_2'], lib="torch", suffix=2)
