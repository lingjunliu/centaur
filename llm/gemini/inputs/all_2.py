
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_all_inputs():
    list_of_inputs = []

    # Case 1: 1D bool tensor
    input1 = np.array([True, True, False, True], dtype=bool)
    input_dict1 = {"input": input1, "dim": None, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: 2D int tensor, dim=0
    input2 = np.array([[1, 1, 1], [0, 1, 1], [1, 1, 0]], dtype=int)
    input_dict2 = {"input": input2, "dim": 0, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 3D float tensor, dim=1, keepdim=True
    input3 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict3 = {"input": input3, "dim": 1, "keepdim": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: 2D bool tensor, dim=1, keepdim=False
    input4 = np.array([[True, True], [False, True], [True, False]], dtype=bool)
    input_dict4 = {"input": input4, "dim": 1, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Case 5: 2D uint8 tensor, dim=0, keepdim=True
    input5 = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=np.uint8)
    input_dict5 = {"input": input5, "dim": 0, "keepdim": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Case 6: 3D int tensor, dim=(0, 1)
    input6 = np.array([[[1, 1], [1, 1]], [[1, 0], [0, 1]]], dtype=int)
    input_dict6 = {"input": input6, "dim": (0, 1), "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Case 7: 1D int tensor with negative values
    input7 = np.array([-1, -2, -3, 0], dtype=int)
    input_dict7 = {"input": input7, "dim": None, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.all_2"] = torch_all_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.all_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.all_2'.")

check_valid('torch.all', generated_inputs['torch.all_2'], lib="torch")
