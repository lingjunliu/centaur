
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def atanh_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([-0.5, 0, 0.5]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Multi-dimensional float tensor with negative values
    input2 = np.array([[-0.9, -0.2, 0.1], [0.3, 0.6, 0.8]]).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D float tensor close to the limit
    input3 = np.array([0.99, -0.99]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Scalar input
    input4 = np.array(0.7).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger multi-dimensional tensor
    input5 = np.random.rand(3, 4, 5).astype(np.float64) * 0.8 - 0.4 # Values between -0.4 and 0.4
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))


    return list_of_inputs

generated_inputs["torch.atanh_"] = atanh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.atanh_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atanh_'.")

check_valid('torch.atanh_', generated_inputs['torch.atanh_'], lib="torch")
