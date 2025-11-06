
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def local_response_norm_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.randn(4, 5, 10, 10, dtype=torch.float32).numpy()
    input_dict = {
        "size": 2,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.randn(2, 4, 16, dtype=torch.float32).numpy()
    input_dict = {
        "size": 5,
        "alpha": 0.001,
        "beta": 0.5,
        "k": 2.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3 (fixed to be 3D)
    input_arr = torch.randn(3, 7, 5, dtype=torch.float32).numpy()
    input_dict = {
        "size": 1,
        "alpha": 0.0,
        "beta": 1.0,
        "k": 0.5,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input_arr = torch.randn(8, 4, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "size": 7,
        "alpha": 1e-5,
        "beta": 0.9,
        "k": 1.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5 (alpha made non-negative)
    input_arr = torch.randn(2, 6, 3, 3, 3, dtype=torch.float64).numpy()
    input_dict = {
        "size": 3,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.5,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = torch.randn(1, 2, 2, 2, 2, 2, dtype=torch.float32).numpy()
    input_dict = {
        "size": 4,
        "alpha": 0.2,
        "beta": 0.25,
        "k": 0.1,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    input_arr = torch.randn(5, 10, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "size": 9,
        "alpha": 1e-5,
        "beta": 2.0,
        "k": 0.9,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input_arr = torch.randn(3, 3, 12, 12, dtype=torch.float32).numpy()
    input_dict = {
        "size": 2,
        "alpha": 0.5,
        "beta": 0.0,
        "k": 1.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = torch.randn(2, 8, 5, dtype=torch.float64).numpy()
    input_dict = {
        "size": np.int64(3),
        "alpha": np.float64(0.001),
        "beta": np.float32(0.75),
        "k": np.float64(1.0),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    input_arr = torch.randn(7, 16, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "size": 5,
        "alpha": 1e-6,
        "beta": 1.5,
        "k": 10.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11 (alpha made non-negative)
    input_arr = torch.randn(1, 1, 20, dtype=torch.float32).numpy()
    input_dict = {
        "size": 2,
        "alpha": 0.5,
        "beta": 0.75,
        "k": 100.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12 (avoid float16 on CPU)
    input_arr = torch.randn(2, 3, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "size": np.int32(3),
        "alpha": np.float32(0.0001),
        "beta": np.float32(0.75),
        "k": np.float32(1.0),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.LocalResponseNorm"] = local_response_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LocalResponseNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LocalResponseNorm'.")


check_valid('torch.nn.LocalResponseNorm', generated_inputs['torch.nn.LocalResponseNorm'], lib="torch", suffix=0)
