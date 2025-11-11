
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def unique_consecutive_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([1, 1, 2, 2, 3, 1, 1, 2])
    input_dict = {
        "input": input_tensor,
        "return_inverse": False,
        "return_counts": False,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([5, 5, 5, 3, 3, 1])
    input_dict = {
        "input": input_tensor,
        "return_inverse": True,
        "return_counts": False,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([10, 10, 20, 20, 20, 30])
    input_dict = {
        "input": input_tensor,
        "return_inverse": False,
        "return_counts": True,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([7, 7, 8, 8, 9, 9, 9])
    input_dict = {
        "input": input_tensor,
        "return_inverse": True,
        "return_counts": True,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1, 2], [1, 2], [3, 4], [3, 4]])
    input_dict = {
        "input": input_tensor,
        "return_inverse": False,
        "return_counts": False,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1, 1, 2, 2], [3, 3, 4, 4]])
    input_dict = {
        "input": input_tensor,
        "return_inverse": False,
        "return_counts": False,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[5, 6], [5, 6], [7, 8]])
    input_dict = {
        "input": input_tensor,
        "return_inverse": True,
        "return_counts": False,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[2, 2, 3], [4, 4, 5]])
    input_dict = {
        "input": input_tensor,
        "return_inverse": False,
        "return_counts": True,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1, 2]], [[1, 2]], [[3, 4]]])
    input_dict = {
        "input": input_tensor,
        "return_inverse": False,
        "return_counts": False,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.5, 1.5, 2.5, 2.5, 3.5])
    input_dict = {
        "input": input_tensor,
        "return_inverse": True,
        "return_counts": True,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.unique_consecutive"] = unique_consecutive_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.unique_consecutive' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unique_consecutive'.")


check_valid('torch.unique_consecutive', generated_inputs['torch.unique_consecutive'], lib="torch", suffix=0)
