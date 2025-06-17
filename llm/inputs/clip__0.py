
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def clip__inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float tensor
    input_tensor = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    min_val = 0.0
    max_val = 2.0
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int tensor with negative min and max
    input_tensor = np.array([[-2, -1, 0], [1, 2, 3]], dtype=np.int32)
    min_val = -1.0
    max_val = 2.0
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float tensor with different min and max
    input_tensor = np.random.rand(2, 3, 4).astype(np.float64)
    min_val = 0.2
    max_val = 0.8
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor with min > max (should be handled correctly)
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    min_val = 3.0
    max_val = 1.0
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    input_tensor = np.array([-5.0, -2.0, 0.0, 3.0], dtype=np.float32)
    min_val = -3.0
    max_val = 1.0
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.clip_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clip_'.")

check_valid('torch.clip_', generated_inputs['torch.clip_'], lib="torch")
