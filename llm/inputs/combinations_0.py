
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def combinations_inputs():
    list_of_inputs = []

    # Test case 1: Basic integer tensor
    input_tensor = np.array([1, 2, 3, 4])
    input_dict = {"input": input_tensor, "r": 2, "with_replacement": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Different r value
    input_tensor = np.array([1, 2, 3])
    input_dict = {"input": input_tensor, "r": 3, "with_replacement": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: with_replacement = True
    input_tensor = np.array([1, 2, 3])
    input_dict = {"input": input_tensor, "r": 2, "with_replacement": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Float tensor
    input_tensor = np.array([1.0, 2.0, 3.0])
    input_dict = {"input": input_tensor, "r": 2, "with_replacement": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Negative values
    input_tensor = np.array([-1, 0, 1])
    input_dict = {"input": input_tensor, "r": 2, "with_replacement": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6: r = 1
    input_tensor = np.array([1, 2, 3])
    input_dict = {"input": input_tensor, "r": 1, "with_replacement": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: Larger tensor and r
    input_tensor = np.array([1, 2, 3, 4, 5])
    input_dict = {"input": input_tensor, "r": 3, "with_replacement": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.combinations"] = combinations_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.combinations' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.combinations'.")

check_valid('torch.combinations', generated_inputs['torch.combinations'], lib="torch")
