
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def sin__inputs():
    list_of_inputs = []

    # Input 1: Float tensor
    input1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor -> Convert to float
    input2 = np.random.randint(-5, 5, size=(2, 2), dtype=np.int32).astype(np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Tensor with negative values
    input3 = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi]).astype(np.float64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor
    input4 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Large tensor
    input5 = np.random.randn(100, 100).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.sin_"] = sin__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sin_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sin_'.")

check_valid('torch.sin_', generated_inputs['torch.sin_'], lib="torch")
