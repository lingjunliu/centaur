
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def sqrt_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor with positive floats
    input1 = torch.randn(4).abs().numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with positive floats
    input2 = torch.rand(2, 3).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor with integers (converted to float)
    input3 = torch.randint(0, 10, (2, 2, 2)).float().numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar tensor
    input4 = np.array(9.0).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: 1D tensor with zeros
    input5 = torch.zeros(5).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.sqrt"] = sqrt_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sqrt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sqrt'.")

check_valid('torch.sqrt', generated_inputs['torch.sqrt'], lib="torch")
