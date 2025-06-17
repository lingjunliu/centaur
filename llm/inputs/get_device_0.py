
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def get_device_inputs():
    list_of_inputs = []

    # Case 1: Float tensor on CPU
    tensor1 = torch.randn(2, 3).numpy()
    input_dict1 = {"obj": tensor1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Int tensor on CPU
    tensor2 = torch.randint(0, 10, (4, 5)).numpy()
    input_dict2 = {"obj": tensor2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Long tensor on CPU, 3D
    tensor3 = torch.randint(-5, 5, (2, 2, 2), dtype=torch.long).numpy()
    input_dict3 = {"obj": tensor3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Bool tensor
    tensor4 = torch.tensor([[True, False], [False, True]]).numpy()
    input_dict4 = {"obj": tensor4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Double tensor
    tensor5 = torch.randn(3, 4, dtype=torch.float64).numpy()
    input_dict5 = {"obj": tensor5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.get_device"] = get_device_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.get_device' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.get_device'.")

check_valid('torch.get_device', generated_inputs['torch.get_device'], lib="torch")
