
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def lt_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1, 2, 3])
    other_float = 2.0
    out_tensor = np.array([False, False, False])

    input_dict = {
        "input": input_tensor,
        "other": other_float,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 2], [3, 4]])
    other_float = 3.0
    out_tensor = np.array([[False, False], [False, False]])
    input_dict = {
        "input": input_tensor,
        "other": other_float,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([-1, -2, -3])
    other_float = 0.0
    out_tensor = np.array([False, False, False])
    input_dict = {
        "input": input_tensor,
        "other": other_float,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1.5, 2.5, 3.5])
    other_float = 2.0
    out_tensor = np.array([False, False, False])
    input_dict = {
        "input": input_tensor,
        "other": other_float,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([1, 2, 3], dtype=np.int64)
    other_float = 2.0
    out_tensor = np.array([False, False, False])
    input_dict = {
        "input": input_tensor,
        "other": other_float,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]])
    other_float = 5.0
    out_tensor = np.array([[False, False, False], [False, False, False]])
    input_dict = {
        "input": input_tensor,
        "other": other_float,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([1, 2, 3, 4, 5])
    other_float = 6.0
    out_tensor = np.array([False, False, False, False, False])
    input_dict = {
        "input": input_tensor,
        "other": other_float,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
    other_float = -2.5
    out_tensor = np.array([False, False, False, False, False])
    input_dict = {
        "input": input_tensor,
        "other": other_float,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = np.array([0])
    other_float = 0.0001
    out_tensor = np.array([False])

    input_dict = {
        "input": input_tensor,
        "other": other_float,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([1000, 2000, 3000])
    other_float = 2500.0
    out_tensor = np.array([False, False, False])

    input_dict = {
        "input": input_tensor,
        "other": other_float,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.array([1, 2, 3, 4]).reshape((2, 2))
    other_float = 2.5
    out_tensor = np.array([[False, False], [False, False]])

    input_dict = {
        "input": input_tensor,
        "other": other_float,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.lt_2"] = lt_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.lt_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lt_2'.")

check_valid('torch.lt', generated_inputs['torch.lt_2'], lib="torch", suffix=2)
