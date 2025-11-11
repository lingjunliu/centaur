
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool3d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with tuple parameters
    input_tensor = torch.randn(20, 16, 50, 44, 31).numpy()
    input_dict = {
        "input": input_tensor,
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Non-square window with different stride
    input_tensor = torch.randn(10, 8, 30, 25, 20).numpy()
    input_dict = {
        "input": input_tensor,
        "kernel_size": (3, 2, 2),
        "stride": (2, 1, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - With padding
    input_tensor = torch.randn(5, 4, 10, 8, 6).numpy()
    input_dict = {
        "input": input_tensor,
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With dilation
    input_tensor = torch.randn(3, 2, 15, 10, 8).numpy()
    input_dict = {
        "input": input_tensor,
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (2, 2, 2),
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With return_indices True
    input_tensor = torch.randn(1, 1, 8, 6, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - With ceil_mode True
    input_tensor = torch.randn(2, 3, 15, 10, 8).numpy()
    input_dict = {
        "input": input_tensor,
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Large kernel size with stride
    input_tensor = torch.randn(1, 1, 20, 15, 10).numpy()
    input_dict = {
        "input": input_tensor,
        "kernel_size": (5, 4, 3),
        "stride": (3, 2, 1),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Different padding values
    input_tensor = torch.randn(1, 1, 10, 8, 6).numpy()
    input_dict = {
        "input": input_tensor,
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": (1, 2, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - With negative values in input tensor
    input_tensor = torch.randn(2, 3, 10, 8, 6).numpy()
    input_tensor[input_tensor < 0] = -1.0
    input_dict = {
        "input": input_tensor,
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - With large dilation values
    input_tensor = torch.randn(1, 1, 30, 25, 20).numpy()
    input_dict = {
        "input": input_tensor,
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (3, 3, 3),
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_2"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_2'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_2'], lib="torch", suffix=2)
