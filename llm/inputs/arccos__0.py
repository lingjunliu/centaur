
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def arccos__inputs():
    list_of_inputs = []

    # Input 1: Simple float tensor
    input1 = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor
    input2 = np.array([[0.2, 0.7], [-0.3, -0.9]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor
    input3 = np.random.uniform(low=-1.0, high=1.0, size=(2, 3, 4)).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with values outside [-1, 1] (should still run without error, producing nan)
    input4 = np.array([-2.0, 0.5, 1.5], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Large tensor
    input5 = np.random.uniform(low=-1.0, high=1.0, size=(100, 100)).astype(np.float64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.arccos_"] = arccos__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.arccos_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arccos_'.")

check_valid('torch.arccos_', generated_inputs['torch.arccos_'], lib="torch")
