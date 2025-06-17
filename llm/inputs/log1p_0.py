
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def log1p_inputs():
    list_of_inputs = []

    # Input 1: Scalar float
    input_1 = np.array(0.5, dtype=np.float32)
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D tensor with positive values
    input_2 = np.array([0.1, 0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D tensor with positive values
    input_3 = np.array([[0.0, 0.2], [0.3, 1.5]], dtype=np.float32)
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D tensor with positive values
    input_4 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict_4 = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Larger values
    input_5 = np.array([10.0, 100.0, 1000.0], dtype=np.float64)
    input_dict_5 = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.log1p"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.log1p'.")

check_valid('torch.log1p', generated_inputs['torch.log1p'], lib="torch")
