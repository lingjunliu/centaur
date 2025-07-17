
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def fold_inputs():
    list_of_inputs = []

    # Input 1
    output_size = (4, 5)
    kernel_size = (2, 2)
    dilation = 1
    padding = 0
    stride = 1
    input_tensor = torch.randn(1, 3 * 2 * 2, 12).numpy()

    input_dict = {
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    output_size = (5, 6)
    kernel_size = (3, 3)
    dilation = 2
    padding = 1
    stride = 2
    input_tensor = torch.randn(1, 3 * 3 * 3, 4).numpy()

    input_dict = {
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    output_size = (10, 11)
    kernel_size = (4, 4)
    dilation = 1
    padding = 2
    stride = 3
    input_tensor = torch.randn(1, 3 * 4 * 4, 4).numpy() #Incorrect L

    input_dict = {
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }
    #list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    output_size = (7, 8)
    kernel_size = (2, 2)
    dilation = (3, 3)
    padding = 0
    stride = 1
    input_tensor = torch.randn(1, 3 * 2 * 2, 20).numpy()

    input_dict = {
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    output_size = (6, 7)
    kernel_size = (3, 3)
    dilation = 1
    padding = 1
    stride = 1
    input_tensor = torch.randn(1, 3 * 3 * 3, 42).numpy()

    input_dict = {
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    output_size = (8, 9)
    kernel_size = (2, 2)
    dilation = 2
    padding = 2
    stride = 2
    input_tensor = torch.randn(1, 3 * 2 * 2, 30).numpy()

    input_dict = {
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Unbatched input
    output_size = (4, 5)
    kernel_size = (2, 2)
    dilation = 1
    padding = 0
    stride = 1
    input_tensor = torch.randn(3 * 2 * 2, 12).numpy() # Unbatched

    input_dict = {
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batched input with different output_size
    output_size = (6, 7)
    kernel_size = (2, 2)
    dilation = 1
    padding = 0
    stride = 1
    input_tensor = torch.randn(2, 3 * 2 * 2, 25).numpy() # Batched

    input_dict = {
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 Different Kernel Size and Padding with Tuple
    output_size = (7, 8)
    kernel_size = (3, 3)
    dilation = 1
    padding = (1, 1)
    stride = 1
    input_tensor = torch.randn(1, 3 * 3 * 3, 36).numpy()
    input_dict = {
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 Kernel Size Tuple Dilation Tuple, Padding Tuple, Stride Tuple
    output_size = (5, 6)
    kernel_size = (3, 3)
    dilation = (2, 2)
    padding = (1, 1)
    stride = (2, 2)
    input_tensor = torch.randn(1, 3 * 3 * 3, 4).numpy() #Incorrect L

    input_dict = {
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }
    #list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Fold_3"] = fold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Fold_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Fold_3'.")

check_valid('torch.nn.Fold', generated_inputs['torch.nn.Fold_3'], lib="torch", suffix=3)
