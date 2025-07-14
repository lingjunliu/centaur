
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def repeat_interleave_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1, 2, 3])
    repeats = 2
    dim = 0

    input_dict = {
        "input": input_tensor,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 2], [3, 4]])
    repeats = 3
    dim = 0

    input_dict = {
        "input": input_tensor,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1, 2], [3, 4]])
    repeats = 2
    dim = 1

    input_dict = {
        "input": input_tensor,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    repeats = 2
    dim = 0

    input_dict = {
        "input": input_tensor,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    repeats = 2
    dim = 1

    input_dict = {
        "input": input_tensor,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    repeats = 2
    dim = 2

    input_dict = {
        "input": input_tensor,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_tensor = np.array([1])
    repeats = 5
    dim = 0

    input_dict = {
        "input": input_tensor,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[1,2,3]])
    repeats = 2
    dim = 1

    input_dict = {
        "input": input_tensor,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = np.array([1,2,3,4,5])
    repeats = 1
    dim = 0

    input_dict = {
        "input": input_tensor,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[1,2,3]]])
    repeats = np.array(1)
    dim = 2

    input_dict = {
        "input": input_tensor,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_tensor = np.array([1, 2, 3], dtype=np.int64)
    repeats = np.array(2, dtype=np.int64)
    dim = 0

    input_dict = {
        "input": input_tensor,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.repeat_interleave_1"] = repeat_interleave_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.repeat_interleave_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.repeat_interleave_1'.")

check_valid('torch.repeat_interleave', generated_inputs['torch.repeat_interleave_1'], lib="torch", suffix=1)
