
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def parameterlist_inputs():
    list_of_inputs = []

    # Input 1: A list with a single 1D tensor.
    input_dict_1 = {'values': [np.array([1.0, 2.0, 3.0], dtype=np.float32)]}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: A list with a single 2D tensor.
    input_dict_2 = {'values': [np.ones((3, 3), dtype=np.float32)]}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: A list with multiple 2D tensors of different shapes.
    input_dict_3 = {'values': [np.zeros((2, 2), dtype=np.float32), np.ones((4, 1), dtype=np.float32)]}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: A longer list of simple 1D tensors.
    input_dict_4 = {'values': [np.arange(5, dtype=np.float32) for i in range(5)]}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: A list containing a 3D tensor.
    input_dict_5 = {'values': [np.random.rand(2, 3, 4).astype(np.float32)]}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: List containing a tensor with a single element.
    input_dict_6 = {'values': [np.array([[42.0]], dtype=np.float32)]}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: List containing a tensor with negative values.
    input_dict_7 = {'values': [np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)]}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: List containing high-rank tensor
    input_dict_8 = {'values': [np.ones((1, 2, 1, 3, 1), dtype=np.float32)]}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: List of tensors with different ranks
    input_dict_9 = {'values': [np.ones(5, dtype=np.float32), np.zeros((2,2), dtype=np.float32)]}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: List with a single large tensor
    input_dict_10 = {'values': [np.random.rand(10, 10).astype(np.float32)]}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.nn.ParameterList"] = parameterlist_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ParameterList' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ParameterList'.")

check_valid('torch.nn.ParameterList', generated_inputs['torch.nn.ParameterList'], lib="torch", suffix=0)
