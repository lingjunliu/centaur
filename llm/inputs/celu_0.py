
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def celu_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, positive alpha, inplace=False
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha_val = 1.0
    inplace_val = False

    input_dict = {
        "input": input_tensor,
        "alpha": alpha_val,
        "inplace": inplace_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor, negative values, positive alpha, inplace=True
    input_tensor = np.array([-1.0, -2.0, -3.0, 0.0, 1.0], dtype=np.float32)
    alpha_val = 2.0
    inplace_val = True

    input_dict = {
        "input": input_tensor,
        "alpha": alpha_val,
        "inplace": inplace_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor, mixed values, small alpha, inplace=False
    input_tensor = np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]], dtype=np.float32)
    alpha_val = 0.5
    inplace_val = False

    input_dict = {
        "input": input_tensor,
        "alpha": alpha_val,
        "inplace": inplace_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor, zero alpha, mixed values, inplace=True
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    alpha_val = 0.0
    inplace_val = True

    input_dict = {
        "input": input_tensor,
        "alpha": alpha_val,
        "inplace": inplace_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensor, large alpha, mixed values, inplace=False
    input_tensor = np.array([-5.0, -2.0, 0.0, 2.0, 5.0], dtype=np.float32)
    alpha_val = 10.0
    inplace_val = False

    input_dict = {
        "input": input_tensor,
        "alpha": alpha_val,
        "inplace": inplace_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.celu"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.celu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.celu'.")

check_valid('torch.celu', generated_inputs['torch.celu'], lib="torch", suffix=0)
