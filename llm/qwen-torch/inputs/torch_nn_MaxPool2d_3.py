
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with square kernel and stride
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Non-square window with stride
    input = torch.randn(10, 8, 60, 40).numpy()
    input_dict = {
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - With padding (valid)
    input = torch.randn(5, 4, 100, 80).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With dilation
    input = torch.randn(3, 2, 50, 30).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With return_indices
    input = torch.randn(1, 1, 20, 15).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - With ceil mode
    input = torch.randn(1, 2, 50, 30).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Different kernel size and stride
    input = torch.randn(2, 4, 100, 80).numpy()
    input_dict = {
        "kernel_size": (5, 4),
        "stride": (3, 2),
        "padding": (2, 2),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Large padding values but valid (max padding should be less than or equal to half of kernel size)
    input = torch.randn(1, 3, 200, 150).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": (1, 1),
        "padding": (2, 2),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Negative values in input tensor
    input = torch.randn(2, 3, 50, 40).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Different dimensions (2D tensor)
    input = torch.randn(3, 5, 60, 40).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_3"] = maxpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool2d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool2d_3'.")


check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_3'], lib="torch", suffix=3)
