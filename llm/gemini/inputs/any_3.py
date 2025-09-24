
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def any_inputs():
    list_of_inputs = []

    # Input 1: Basic boolean tensor
    input1 = np.array([[False, True], [False, False]])
    input_dict1 = {"input": input1, "dim": None, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor with dim and keepdim
    input2 = np.array([[0, 1, 0], [0, 0, 2]])
    input_dict2 = {"input": input2, "dim": (0,), "keepdim": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor with negative values and dim
    input3 = np.array([[-1.0, 0.0, 2.0], [-0.5, 0.0, 1.5]])
    input_dict3 = {"input": input3, "dim": (1,), "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor
    input4 = np.random.rand(2, 3, 4)
    input4 = input4 < 0.5
    input_dict4 = {"input": input4, "dim": (0, 1), "keepdim": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D tensor
    input5 = np.array([0, 0, 1, 0])
    input_dict5 = {"input": input5, "dim": None, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: empty tensor
    input6 = np.array([])
    input_dict6 = {"input": input6, "dim": None, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: uint8 tensor
    input7 = np.array([0, 1, 2], dtype=np.uint8)
    input_dict7 = {"input": input7, "dim": None, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: boolean tensor with all False values
    input8 = np.array([[False, False], [False, False]])
    input_dict8 = {"input": input8, "dim": None, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: multi dim
    input9 = np.random.rand(2, 3, 4, 5)
    input9 = input9 < 0.5
    input_dict9 = {"input": input9, "dim": (0, 2), "keepdim": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    return list_of_inputs

generated_inputs["torch.any_3"] = any_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.any_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.any_3'.")

check_valid('torch.any', generated_inputs['torch.any_3'], lib="torch")
