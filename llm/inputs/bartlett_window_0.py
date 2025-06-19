
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def bartlett_window_inputs():
    list_of_inputs = []

    # Input 1: Basic case with n as int
    input_dict = {
        "n": np.int32(5),
        "periodic": False,
        "dtype": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: n as int with periodic=True
    input_dict = {
        "n": np.int64(10),
        "periodic": True,
        "dtype": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: n as int with dtype=torch.float64
    input_dict = {
        "n": np.int32(7),
        "periodic": False,
        "dtype": torch.float64,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: n as int with requires_grad=True
    input_dict = {
        "n": np.int64(12),
        "periodic": True,
        "dtype": None,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: n as int with specific dtype and requires_grad
    input_dict = {
        "n": np.int32(9),
        "periodic": False,
        "dtype": torch.float32,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: large n
    input_dict = {
        "n": np.int64(256),
        "periodic": False,
        "dtype": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.bartlett_window"] = bartlett_window_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bartlett_window' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bartlett_window'.")

check_valid('torch.bartlett_window', generated_inputs['torch.bartlett_window'], lib="torch")
