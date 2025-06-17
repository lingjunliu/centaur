
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_exp_inputs():
    list_of_inputs = []

    # Example 1: 1D tensor with positive and negative values
    input_tensor = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 2D tensor with floating-point values
    input_tensor = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 3D tensor with positive values
    input_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Scalar value
    input_tensor = np.array(0.5, dtype=np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Larger 1D tensor with negative values
    input_tensor = np.random.randn(10).astype(np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: Empty tensor
    input_tensor = np.array([], dtype=np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.exp"] = torch_exp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.exp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.exp'.")

check_valid('torch.exp', generated_inputs['torch.exp'], lib="torch")
