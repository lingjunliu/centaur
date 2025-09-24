
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def tan__inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative values
    input2 = np.random.uniform(-5, 0, size=(2, 2)).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different shape
    input3 = np.random.randn(1, 5, 5, 1).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Small values around 0
    input4 = np.array([-0.1, 0, 0.1]).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Int tensor -> convert to float
    input5 = np.random.randint(-3, 3, size=(2, 3)).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.tan_"] = tan__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.tan_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tan_'.")

check_valid('torch.tan_', generated_inputs['torch.tan_'], lib="torch")
