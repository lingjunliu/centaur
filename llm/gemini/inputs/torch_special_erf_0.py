
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def erf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input_tensor = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    out_tensor = np.array([], dtype=np.float32)

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    input_tensor = np.array([[0.5, -0.5], [1.5, -1.5]], dtype=np.float64)
    out_tensor = np.array([], dtype=np.float64)

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    input_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    out_tensor = np.array([], dtype=np.float32)

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with zeros
    input_tensor = np.zeros((5, 5), dtype=np.float32)
    out_tensor = np.array([], dtype=np.float32)

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with ones
    input_tensor = np.ones((3, 3), dtype=np.float64)
    out_tensor = np.array([], dtype=np.float64)

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with large values
    input_tensor = np.array([100.0, -100.0, 1000.0, -1000.0], dtype=np.float32)
    out_tensor = np.array([], dtype=np.float32)

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with small values
    input_tensor = np.array([0.001, -0.001, 0.0001, -0.0001], dtype=np.float64)
    out_tensor = np.array([], dtype=np.float64)

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with mixed values
    input_tensor = np.array([0.0, 1.0, -1.0, 10.0, -10.0, 0.1, -0.1], dtype=np.float32)
    out_tensor = np.array([], dtype=np.float32)

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with a specific shape
    input_tensor = np.random.rand(1, 5, 1).astype(np.float64)
    out_tensor = np.array([], dtype=np.float64)

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with negative and positive values
    input_tensor = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.float32)
    out_tensor = np.array([], dtype=np.float32)

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.erf"] = erf_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.erf'.")

check_valid('torch.special.erf', generated_inputs['torch.special.erf'], lib="torch", suffix=0)
