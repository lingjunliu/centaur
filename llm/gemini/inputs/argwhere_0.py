
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def argwhere_inputs():
    list_of_inputs = []

    # Example 1: 1D tensor with positive integers
    input_1 = np.array([1, 0, 2, 0, 3])
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Example 2: 2D tensor with mixed positive and negative integers
    input_2 = np.array([[1, -2, 0], [0, 3, -4]])
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Example 3: 3D tensor with floats
    input_3 = np.array([[[0.1, 0.0, 0.2], [0.0, 0.3, 0.4]], [[0.5, 0.6, 0.0], [0.7, 0.0, 0.9]]])
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Example 4: 4D tensor with boolean values
    input_4 = np.array([[[[True, False], [False, True]], [[False, True], [True, False]]], [[[True, True], [False, False]], [[False, False], [True, True]]]])
    input_dict_4 = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Example 5: 2D tensor with complex numbers
    input_5 = np.array([[1+1j, 0+0j, 2-2j], [0+0j, 3+0j, 4-1j]])
    input_dict_5 = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Example 6: 1D tensor with only zeros
    input_6 = np.array([0, 0, 0, 0])
    input_dict_6 = {"input": input_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Example 7: 5D tensor with a mix of positive and negative integers
    input_7 = np.random.randint(-5, 5, size=(2, 2, 2, 2, 2))
    input_dict_7 = {"input": input_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.argwhere"] = argwhere_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.argwhere' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.argwhere'.")

check_valid('torch.argwhere', generated_inputs['torch.argwhere'], lib="torch")
