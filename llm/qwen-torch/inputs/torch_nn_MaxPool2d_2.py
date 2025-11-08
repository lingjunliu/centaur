
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with square kernel and stride
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Non-square kernel and stride
    input = torch.randn(10, 8, 64, 32).numpy()
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
    
    # Input 3 - With padding
    input = torch.randn(5, 4, 100, 50).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With dilation
    input = torch.randn(5, 4, 20, 20).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With return_indices True
    input = torch.randn(1, 2, 10, 10).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - With ceil_mode True
    input = torch.randn(1, 1, 10, 10).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Mixed parameters
    input = torch.randn(3, 4, 256, 128).numpy()
    input_dict = {
        "kernel_size": (4, 4),
        "stride": (2, 2),
        "padding": (2, 2),
        "dilation": (2, 2),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Negative values in input tensor
    input = torch.randn(1, 2, 10, 10).numpy()
    input[0, 0, 0, 0] = -1.0
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Different dimensions in input tensor
    input = torch.randn(2, 4, 64, 32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Large input tensor
    input = torch.randn(2, 8, 1024, 512).numpy()
    input_dict = {
        "kernel_size": (5, 5),
        "stride": (3, 3),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_2"] = maxpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool2d_2'.")


check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_2'], lib="torch", suffix=2)
