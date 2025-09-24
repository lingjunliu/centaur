
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cosh__inputs():
    list_of_inputs = []

    # Input 1: Float Tensor, 1D
    input1 = np.array([-1.0, 0.0, 1.0, 2.0, -2.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float Tensor, 2D
    input2 = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float Tensor, 3D
    input3 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 5: Float Tensor, scalar
    input5 = np.array(3.14159, dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Float Tensor, with large values to test overflow
    input6 = np.array([-10.0, 0.0, 10.0, 20.0, -20.0], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.cosh_"] = cosh__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cosh_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cosh_'.")

check_valid('torch.cosh_', generated_inputs['torch.cosh_'], lib="torch")
