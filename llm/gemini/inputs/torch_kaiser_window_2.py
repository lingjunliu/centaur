
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def kaiser_window_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "length": np.int32(10),
        "beta": np.float32(0.5),
        "periodic": np.bool_(True),
        "dtype": torch.float32,
        "layout": "strided",
        "requires_grad": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "length": np.int32(5),
        "beta": np.float32(2.0),
        "periodic": np.bool_(False),
        "dtype": torch.float64,
        "layout": "strided",
        "requires_grad": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "length": np.int32(20),
        "beta": np.float32(5.0),
        "periodic": np.bool_(True),
        "dtype": torch.float32,
        "layout": "strided",
        "requires_grad": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "length": np.int32(3),
        "beta": np.float32(0.0),
        "periodic": np.bool_(False),
        "dtype": torch.float32,
        "layout": "strided",
        "requires_grad": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "length": np.int32(15),
        "beta": np.float32(8.0),
        "periodic": np.bool_(True),
        "dtype": torch.float64,
        "layout": "strided",
        "requires_grad": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "length": np.int32(7),
        "beta": np.float32(0.1),
        "periodic": np.bool_(False),
        "dtype": torch.float64,
        "layout": "strided",
        "requires_grad": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "length": np.int32(32),
        "beta": np.float32(12.0),
        "periodic": np.bool_(True),
        "dtype": torch.float32,
        "layout": "strided",
        "requires_grad": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "length": np.int32(64),
        "beta": np.float32(20.0),
        "periodic": np.bool_(False),
        "dtype": torch.float64,
        "layout": "strided",
        "requires_grad": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "length": np.int32(1),
        "beta": np.float32(0.001),
        "periodic": np.bool_(True),
        "dtype": torch.float32,
        "layout": "strided",
        "requires_grad": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "length": np.int32(128),
        "beta": np.float32(10.0),
        "periodic": np.bool_(False),
        "dtype": torch.float64,
        "layout": "strided",
        "requires_grad": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
def convert_to_torch_type(input_dict):
    new_dict = {}
    for key, value in input_dict.items():
        if key == "dtype":
            new_dict[key] = value
        elif key == "length":
            new_dict[key] = int(value)
        elif key == "beta":
            new_dict[key] = float(value)
        elif key == "periodic":
            new_dict[key] = bool(value)
        elif key == "requires_grad":
            new_dict[key] = bool(value)
        elif key == "layout":
            new_dict[key] = None # layout can be None
        else:
            new_dict[key] = value
    return new_dict

generated_inputs["torch.kaiser_window_2"] = [convert_to_torch_type(item) for item in kaiser_window_inputs()]

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.kaiser_window_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.kaiser_window_2'.")

check_valid('torch.kaiser_window', generated_inputs['torch.kaiser_window_2'], lib="torch", suffix=2)
