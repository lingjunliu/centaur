
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def exp2_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    input = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    out = np.array([], dtype=np.float32)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative values
    input = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
    out = np.array([], dtype=np.float64)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array
    input = np.array([[0, 1], [2, 3]], dtype=np.float32)
    out = np.array([], dtype=np.float32)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large values
    input = np.array([10, 20, 30], dtype=np.float64)
    out = np.array([], dtype=np.float64)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small values (close to zero)
    input = np.array([0.1, 0.01, 0.001], dtype=np.float32)
    out = np.array([], dtype=np.float32)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array
    input = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float64)
    out = np.array([], dtype=np.float64)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed positive and negative
    input = np.array([-1.5, 0.5, 2.5], dtype=np.float32)
    out = np.array([], dtype=np.float32)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zeros only
    input = np.array([0, 0, 0], dtype=np.float64)
    out = np.array([], dtype=np.float64)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Ones only
    input = np.array([1, 1, 1], dtype=np.float32)
    out = np.array([], dtype=np.float32)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: np.int64 array
    input = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    out = np.array([], dtype=np.float64)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: np.int32 array
    input = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    out = np.array([], dtype=np.float32)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.exp2"] = exp2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.exp2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.exp2'.")

check_valid('torch.special.exp2', generated_inputs['torch.special.exp2'], lib="torch", suffix=0)
