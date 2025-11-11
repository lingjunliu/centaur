
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool3d_inputs():
    list_of_inputs = []
    
    # Input 1 - Simple case with integer kernel_size
    input = torch.randn(20, 16, 50, 44, 31).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Non-square window with tuple kernel_size
    input = torch.randn(10, 8, 30, 25, 20).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": (2, 1, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - With padding
    input = torch.randn(5, 4, 20, 15, 10).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With dilation
    input = torch.randn(3, 2, 15, 10, 8).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With return_indices
    input = torch.randn(2, 1, 10, 8, 6).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - With ceil_mode
    input = torch.randn(4, 3, 25, 20, 15).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Negative padding (invalid but valid for the purpose of testing)
    input = torch.randn(6, 5, 30, 25, 20).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": -1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - With all parameters including negative values
    input = torch.randn(8, 7, 40, 35, 30).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 1,
        "dilation": -1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Very large kernel size
    input = torch.randn(1, 1, 100, 80, 60).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Different stride values
    input = torch.randn(15, 10, 35, 30, 25).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 3,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_1"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_1'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_1'], lib="torch", suffix=1)
