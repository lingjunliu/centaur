
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def isneginf_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor with negative infinity
    input1 = np.array([-float('inf'), 1.0, 2.0, -float('inf')], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor with a mix of values
    input2 = np.array([[float('inf'), -float('inf'), 0.0], [1.0, -1.0, -float('inf')]], dtype=np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor, all finite values
    input3 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar float tensor with negative infinity
    input4 = np.array(-float('inf'), dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D integer tensor (should be cast to float)
    input5 = np.array([-1, 0, 1, -2], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.isneginf"] = isneginf_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.isneginf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isneginf'.")

check_valid('torch.isneginf', generated_inputs['torch.isneginf'], lib="torch")
