
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def rad2deg_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor with negative values
    input2 = np.array([[-np.pi, -np.pi/2], [0, np.pi/4]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor
    input3 = np.random.rand(2, 3, 4).astype(np.float32) * np.pi
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Scalar float
    input4 = np.array(np.pi / 6, dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D int tensor (will be cast to float)
    input5 = np.array([0, 1, 2, 3], dtype=np.int32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.rad2deg"] = rad2deg_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.rad2deg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rad2deg'.")

check_valid('torch.rad2deg', generated_inputs['torch.rad2deg'], lib="torch")
