
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def acos_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor within the range [-1, 1]
    input1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor with a different shape (2D)
    input2 = np.array([[-0.8, -0.2], [0.3, 0.9]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor with a different shape (3D)
    input3 = np.random.uniform(low=-1.0, high=1.0, size=(2, 2, 2)).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with values outside the range, expecting NaNs
    input4 = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Empty tensor
    input5 = np.array([], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Float16 tensor
    input6 = np.array([-0.7, 0.1, 0.6], dtype=np.float16)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Scalar value within range.
    input7 = np.array(0.4, dtype=np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.acos_"] = acos_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.acos_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.acos_'.")

check_valid('torch.acos_', generated_inputs['torch.acos_'], lib="torch")
