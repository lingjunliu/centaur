
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def asin_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor within [-1, 1]
    input1 = np.array([-0.5, 0, 0.5], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Multi-dimensional float tensor
    input2 = np.array([[-0.8, 0.2], [0.9, -0.1]], dtype=np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Zero tensor
    input3 = np.zeros((2, 3), dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: One tensor
    input4 = np.ones((1, 2), dtype=np.float64) * 0.5
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Scalar value
    input5 = np.array(0.25, dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.asin"] = asin_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.asin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.asin'.")

check_valid('torch.asin', generated_inputs['torch.asin'], lib="torch")
