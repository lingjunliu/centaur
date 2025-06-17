
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def deg2rad_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, 1D
    input1 = np.array([0.0, 30.0, 45.0, 60.0, 90.0, 180.0, 360.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor, 2D, negative values
    input2 = np.array([[-180, -90], [0, 90], [180, 360]], dtype=np.int32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor, 3D
    input3 = np.array([[[0.0, 45.0], [90.0, 180.0]], [[270.0, 360.0], [450.0, 540.0]]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Float tensor, scalar
    input4 = np.array(270.0, dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Int tensor, empty tensor
    input5 = np.array([], dtype=np.int32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.deg2rad"] = deg2rad_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.deg2rad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.deg2rad'.")

check_valid('torch.deg2rad', generated_inputs['torch.deg2rad'], lib="torch")
