
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def max_pool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensor
    input = torch.randn(1, 1, 5, 5).numpy()
    kernel_size = (2, 2)
    stride = 2
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor
    input = torch.randint(0, 10, (1, 1, 5, 5)).numpy()
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different kernel_size, stride, and padding
    input = torch.randn(1, 3, 10, 10).numpy()
    kernel_size = (4, 4)
    stride = 2
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With ceil_mode
    input = torch.randn(1, 1, 7, 7).numpy()
    kernel_size = (2, 2)
    stride = 3
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple channels and batch size
    input = torch.randn(2, 4, 8, 8).numpy()
    kernel_size = (2, 2)
    stride = 2
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool2d_4"] = max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.max_pool2d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool2d_4'.")

check_valid('torch.nn.functional.max_pool2d', generated_inputs['torch.nn.functional.max_pool2d_4'], lib="torch")
