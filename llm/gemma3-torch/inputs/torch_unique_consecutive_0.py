
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def unique_consecutive_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 1, 2, 2, 3, 4, 4, 4, 5])
    input_dict1 = {
        "input": input1,
        "return_inverse": True,
        "return_counts": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([1, 2, 3, 4, 5])
    input_dict2 = {
        "input": input2,
        "return_inverse": False,
        "return_counts": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([])
    input_dict3 = {
        "input": input3,
        "return_inverse": True,
        "return_counts": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1, 1, 1, 1, 1])
    input_dict4 = {
        "input": input4,
        "return_inverse": True,
        "return_counts": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([5, 4, 3, 2, 1])
    input_dict5 = {
        "input": input5,
        "return_inverse": False,
        "return_counts": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1, 2, 1, 2, 1, 2])
    input_dict6 = {
        "input": input6,
        "return_inverse": True,
        "return_counts": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1, -1, 1, -1, 1])
    input_dict7 = {
        "input": input7,
        "return_inverse": True,
        "return_counts": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1, 1], [2, 2], [3, 3]])
    input_dict8 = {
        "input": input8,
        "return_inverse": False,
        "return_counts": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1, 2], [1, 2], [3, 4]])
    input_dict9 = {
        "input": input9,
        "return_inverse": True,
        "return_counts": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    input_dict10 = {
        "input": input10,
        "return_inverse": True,
        "return_counts": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

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
