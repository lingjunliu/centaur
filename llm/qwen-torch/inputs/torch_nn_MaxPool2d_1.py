
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool2d_inputs():
    list_of_inputs = []
    
    # Input 1 - valid
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - valid
    input = torch.randn(10, 8, 25, 16).numpy()
    input_dict = {
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - valid
    input = torch.randn(5, 4, 10, 8).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - valid
    input = torch.randn(3, 2, 15, 12).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 3,
        "padding": 2,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - valid
    input = torch.randn(1, 1, 50, 32).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - valid
    input = torch.randn(30, 16, 60, 40).numpy()
    input_dict = {
        "kernel_size": (5, 3),
        "stride": (2, 1),
        "padding": 2,
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - valid
    input = torch.randn(15, 8, 20, 10).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - valid
    input = torch.randn(10, 4, 15, 8).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - valid
    input = torch.randn(5, 2, 30, 20).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - valid
    input = torch.randn(25, 16, 40, 30).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": 2,
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_1"] = maxpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool2d_1'.")


check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_1'], lib="torch", suffix=1)
