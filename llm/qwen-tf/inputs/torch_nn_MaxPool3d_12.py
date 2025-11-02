
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool3d_inputs():
    list_of_inputs = []
    
    # 1
    input = torch.randn(2, 3, 8, 10, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 2
    input = torch.randn(4, 7, 9, 11, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": 1,
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 3 (fixed padding)
    input = torch.randn(1, 1, 10, 10, 10, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": 1,
        "padding": (1, 1, 1),
        "dilation": (2, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 4
    input = torch.randn(1, 2, 11, 9, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 3,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 5
    input = torch.randn(3, 3, 5, 4, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 3, 2),
        "stride": 2,
        "padding": (1, 0, 1),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 6
    input = torch.randn(2, 1, 20, 15, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (4, 3, 2),
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": (2, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 7 (4D input)
    input = torch.linspace(-10, 10, steps=2*5*5*5, dtype=torch.float32).reshape(2, 5, 5, 5).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 8 (avoid float16 on CPU)
    input = torch.randn(4, 4, 6, 8, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2, 4),
        "stride": 1,
        "padding": (0, 1, 2),
        "dilation": (1, 2, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 9
    input = torch.randn(1, 3, 9, 9, 9, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (5, 5, 5),
        "stride": 4,
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 10
    input = torch.randn(2, 2, 4, 6, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (4, 6, 2),
        "stride": 2,
        "padding": (2, 3, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 11 (4D input)
    input = torch.randn(1, 8, 7, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": 3,
        "padding": (0, 1, 1),
        "dilation": (1, 2, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 12
    input = torch.randn(1, 4, 3, 3, 20, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 7),
        "stride": 5,
        "padding": (0, 0, 3),
        "dilation": (1, 1, 2),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_12"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_12' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_12'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_12'], lib="torch", suffix=12)
