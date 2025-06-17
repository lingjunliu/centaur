
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def nonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D int tensor
    input_1 = np.array([1, 1, 1, 0, 1])
    input_dict_1 = {"input": input_1, "out": None, "as_tuple": False}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D float tensor
    input_2 = np.array([[0.6, 0.0, 0.0, 0.0],
                        [0.0, 0.4, 0.0, 0.0],
                        [0.0, 0.0, 1.2, 0.0],
                        [0.0, 0.0, 0.0, -0.4]])
    input_dict_2 = {"input": input_2, "out": None, "as_tuple": False}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 1D int tensor with as_tuple=True
    input_3 = np.array([1, 1, 1, 0, 1])
    input_dict_3 = {"input": input_3, "out": None, "as_tuple": True}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2D float tensor with as_tuple=True
    input_4 = np.array([[0.6, 0.0, 0.0, 0.0],
                        [0.0, 0.4, 0.0, 0.0],
                        [0.0, 0.0, 1.2, 0.0],
                        [0.0, 0.0, 0.0, -0.4]])
    input_dict_4 = {"input": input_4, "out": None, "as_tuple": True}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 0D int tensor (scalar) with as_tuple=True
    input_5 = np.array(5)
    input_dict_5 = {"input": input_5, "out": None, "as_tuple": True}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 3D int tensor
    input_6 = np.array([[[1, 0, 1], [0, 1, 0]], [[1, 1, 0], [0, 0, 1]]])
    input_dict_6 = {"input": input_6, "out": None, "as_tuple": False}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: 2D bool tensor
    input_7 = np.array([[True, False, True], [False, True, False]])
    input_dict_7 = {"input": input_7, "out": None, "as_tuple": False}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 1D tensor with negative values
    input_8 = np.array([-1, 0, 1, -2, 2])
    input_dict_8 = {"input": input_8, "out": None, "as_tuple": False}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nonzero"] = nonzero_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nonzero' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nonzero'.")

check_valid('torch.nonzero', generated_inputs['torch.nonzero'], lib="torch")
