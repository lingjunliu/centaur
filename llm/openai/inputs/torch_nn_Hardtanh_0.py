
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def hardtanh_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([-2.5, -0.5, 0.0, 0.5, 3.0], dtype=np.float32)
    input_dict = {
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False,
        "min_value": -1.0,
        "max_value": 1.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = np.random.randn(2, 3).astype(np.float64)
    input_dict = {
        "min_val": -2.0,
        "max_val": 2.0,
        "inplace": True,
        "min_value": -2.0,
        "max_value": 2.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = np.linspace(-3, 7, num=10, dtype=np.float32)
    input_dict = {
        "min_val": 0.0,
        "max_val": 6.0,
        "inplace": False,
        "min_value": 0.0,
        "max_value": 6.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = np.array([[-10.0, -2.0, -0.5],
                          [-1.1, -0.9, -1.5]], dtype=np.float32)
    input_dict = {
        "min_val": -5.0,
        "max_val": -1.0,
        "inplace": False,
        "min_value": -5.0,
        "max_value": -1.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = np.random.uniform(-5, 5, size=(1, 1, 4)).astype(np.float32)
    input_dict = {
        "min_val": -3.5,
        "max_val": -0.5,
        "inplace": True,
        "min_value": -3.5,
        "max_value": -0.5,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = (np.random.randn(3, 2, 2).astype(np.float64) * 1e7)
    input_dict = {
        "min_val": -1e6,
        "max_val": 1e6,
        "inplace": False,
        "min_value": -1e6,
        "max_value": 1e6,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = np.array(0.2, dtype=np.float64)
    input_dict = {
        "min_val": -0.5,
        "max_val": 0.5,
        "inplace": True,
        "min_value": -0.5,
        "max_value": 0.5,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = np.empty((0, 5), dtype=np.float32)
    input_dict = {
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False,
        "min_value": -1.0,
        "max_value": 1.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = np.random.randn(2, 3, 4, 4).astype(np.float32)
    input_dict = {
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": True,
        "min_value": -1.0,
        "max_value": 1.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = np.random.uniform(-3, 3, size=(4, 5, 6)).astype(np.float16)
    input_dict = {
        "min_val": -2.5,
        "max_val": -0.5,
        "inplace": False,
        "min_value": -2.5,
        "max_value": -0.5,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = np.array([np.nan, np.inf, -np.inf, 0.5, -0.5], dtype=np.float64)
    input_dict = {
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False,
        "min_value": -1.0,
        "max_value": 1.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = np.random.randn(1, 2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "min_val": -3.0,
        "max_val": 3.0,
        "inplace": True,
        "min_value": -3.0,
        "max_value": 3.0,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.Hardtanh"] = hardtanh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Hardtanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Hardtanh'.")


check_valid('torch.nn.Hardtanh', generated_inputs['torch.nn.Hardtanh'], lib="torch", suffix=0)
