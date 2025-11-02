
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool3d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.randn(2, 3, 6, 8, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.randn(3, 5, 7, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = torch.randn(1, 1, 10, 12, 14, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 2,
        "padding": (1, 0, 2),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input_arr = torch.randn(4, 2, 15, 17, 19, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 3,
        "padding": (2, 2, 2),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = torch.randn(2, 4, 5, 8, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": (0, 1, 1),
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = torch.randn(3, 3, 7, 9, 11, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    vals = torch.arange(1*2*6*6*6, dtype=torch.float64) - 500.0
    input_arr = vals.view(1, 2, 6, 6, 6).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": 1,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input_arr = torch.randn(1, 5, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 3,
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = torch.randn(1, 1, 20, 10, 16, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 7,
        "stride": 1,
        "padding": (3, 0, 2),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10 (adjusted padding to satisfy constraints)
    input_arr = torch.randn(2, 2, 15, 14, 13, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 3,
        "padding": (0, 1, 0),
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = torch.randn(3, 1, 4, 3, 2, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    vals = torch.linspace(-10.0, 10.0, steps=8*3*3*3, dtype=torch.float32)
    input_arr = vals.view(8, 3, 3, 3).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_3"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_3'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_3'], lib="torch", suffix=3)
