
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def lppool2d_inputs():
    list_of_inputs = []

    input_dict = {
        "norm_type": 2.0,
        "kernel_size": 3,
        "stride": 2,
        "ceil_mode": False,
        "input": np.random.randn(1, 1, 8, 8).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "norm_type": 1.0,
        "kernel_size": 2,
        "stride": 2,
        "ceil_mode": False,
        "input": np.random.randn(2, 3, 10, 12).astype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "norm_type": 3.0,
        "kernel_size": 3,
        "stride": 1,
        "ceil_mode": False,
        "input": np.random.randn(3, 9, 9).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "norm_type": 1.5,
        "kernel_size": 2,
        "stride": 2,
        "ceil_mode": True,
        "input": np.random.randn(1, 5, 7).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "norm_type": float("inf"),
        "kernel_size": 5,
        "stride": 5,
        "ceil_mode": False,
        "input": np.random.randn(4, 2, 15, 15).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "norm_type": 2.5,
        "kernel_size": 2,
        "stride": 3,
        "ceil_mode": False,
        "input": np.random.randn(1, 4, 6, 5).astype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "norm_type": 1.0,
        "kernel_size": 4,
        "stride": 4,
        "ceil_mode": True,
        "input": np.random.randn(8, 1, 4, 4).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "norm_type": 4.0,
        "kernel_size": 1,
        "stride": 1,
        "ceil_mode": False,
        "input": np.random.randn(2, 8, 8).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "norm_type": 2.0,
        "kernel_size": 3,
        "stride": 4,
        "ceil_mode": False,
        "input": np.random.randn(3, 3, 9, 13).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "norm_type": 5.0,
        "kernel_size": 2,
        "stride": 1,
        "ceil_mode": True,
        "input": np.random.randn(5, 6, 7, 7).astype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "norm_type": 10.0,
        "kernel_size": 3,
        "stride": 3,
        "ceil_mode": False,
        "input": np.random.randn(6, 3, 10).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "norm_type": 1.2,
        "kernel_size": 2,
        "stride": 2,
        "ceil_mode": False,
        "input": np.random.randn(1, 1, 2, 3).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.LPPool2d_1"] = lppool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LPPool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LPPool2d_1'.")


check_valid('torch.nn.LPPool2d', generated_inputs['torch.nn.LPPool2d_1'], lib="torch", suffix=1)
