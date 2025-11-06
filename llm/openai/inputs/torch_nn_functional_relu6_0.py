
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def relu6_inputs():
    list_of_inputs = []

    input_dict = {
        "input": np.array([-3.5, 0.0, 2.3, 6.0, 7.8], dtype=np.float32),
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[-10.0, -1.0, 0.0],
                           [0.1, 5.5, 9.9]], dtype=np.float64),
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[[-7.0, -0.5, 0.0],
                            [1.2, 6.1, 8.0]]], dtype=np.float16),
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.random.uniform(-5, 10, size=(2, 3, 4, 5)).astype(np.float32),
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array(7.5, dtype=np.float32),
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([], dtype=np.float32),
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    input_dict = {
        "input": a.T,  # non-contiguous view
        "inplace": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([1000.0, -1000.0, 6.0001, -0.0001], dtype=np.float32),
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.random.randn(1, 2, 3, 4, 5).astype(np.float64),
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([np.nan, np.inf, -np.inf, 5.9, -2.2], dtype=np.float32),
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([-1e-7, 0.0, 1e-7, 6.0, 6.0000005], dtype=np.float32),
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.linspace(-12.0, 12.0, num=24, dtype=np.float16).reshape(2, 3, 4),
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.relu6"] = relu6_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.relu6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.relu6'.")


check_valid('torch.nn.functional.relu6', generated_inputs['torch.nn.functional.relu6'], lib="torch", suffix=0)
