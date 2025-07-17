
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
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)
    input_tensor = torch.randn(1, 3 * 2 * 2, 12).numpy()
    input_dict = {"output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    output_size = (10, 10)
    kernel_size = (3, 3)
    dilation = (2, 2)
    padding = (1, 1)
    stride = (2, 2)
    L = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1) * ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)
    input_tensor = torch.randn(1, 1 * 3 * 3, int(L)).numpy()
    input_dict = {"output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    output_size = (7, 8)
    kernel_size = (2, 3)
    dilation = (1, 1)
    padding = (1, 0)
    stride = (1, 2)
    L = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1) * ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)
    input_tensor = torch.randn(1, 5 * 2 * 3, int(L)).numpy()
    input_dict = {"output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    output_size = (5, 5)
    kernel_size = (1, 1)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)
    L = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1) * ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)
    input_tensor = torch.randn(1, 2 * 1 * 1, int(L)).numpy()
    input_dict = {"output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    output_size = (12, 15)
    kernel_size = (4, 4)
    dilation = (1, 1)
    padding = (2, 2)
    stride = (3, 3)
    L = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1) * ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)
    input_tensor = torch.randn(1, 4 * 4 * 4, int(L)).numpy()
    input_dict = {"output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - No batch dimension
    output_size = (6, 7)
    kernel_size = (2, 2)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)
    L = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1) * ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)
    input_tensor = torch.randn(3 * 2 * 2, int(L)).numpy()
    input_dict = {"output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Different padding and stride
    output_size = (8, 9)
    kernel_size = (3, 3)
    dilation = (1, 1)
    padding = (2, 1)
    stride = (2, 1)
    L = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1) * ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)
    input_tensor = torch.randn(1, 1 * 3 * 3, int(L)).numpy()
    input_dict = {"output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - Larger kernel and output size
    output_size = (20, 20)
    kernel_size = (5, 5)
    dilation = (1, 1)
    padding = (2, 2)
    stride = (1, 1)
    L = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1) * ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)
    input_tensor = torch.randn(1, 2 * 5 * 5, int(L)).numpy()
    input_dict = {"output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Small output size and kernel size
    output_size = (2, 2)
    kernel_size = (1, 1)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)
    L = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1) * ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)
    input_tensor = torch.randn(1, 1 * 1 * 1, int(L)).numpy()
    input_dict = {"output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - dilation > 1
    output_size = (8, 8)
    kernel_size = (3, 3)
    dilation = (2, 2)
    padding = (0, 0)
    stride = (1, 1)
    L = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1) * ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)
    input_tensor = torch.randn(1, 1 * 3 * 3, int(L)).numpy()
    input_dict = {"output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Fold_2"] = fold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Fold_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Fold_2'.")

check_valid('torch.nn.Fold', generated_inputs['torch.nn.Fold_2'], lib="torch", suffix=2)
