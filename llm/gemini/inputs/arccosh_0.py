
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def arccosh_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input_tensor = np.array([1.0, 2.0, 3.0])
    out_tensor = np.array([0.0, 0.0, 0.0])

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    input_tensor = np.array([[1.0, 1.5], [2.0, 2.5]])
    out_tensor = np.array([[0.0, 0.0], [0.0, 0.0]])

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    input_tensor = np.array([[[1.0, 1.1], [1.2, 1.3]], [[2.0, 2.1], [2.2, 2.3]]])
    out_tensor = np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with larger values
    input_tensor = np.array([5.0, 10.0, 15.0])
    out_tensor = np.array([0.0, 0.0, 0.0])

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with float64 dtype
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    out_tensor = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with different shape
    input_tensor = np.array([[2.5, 3.5, 4.5], [5.5, 6.5, 7.5]])
    out_tensor = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.arccosh"] = arccosh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.arccosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arccosh'.")

check_valid('torch.arccosh', generated_inputs['torch.arccosh'], lib="torch", suffix=0)
