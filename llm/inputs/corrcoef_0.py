
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def corrcoef_inputs():
    list_of_inputs = []

    # Example 1: 2D tensor with integer values
    x = np.array([[0, 1, 2], [2, 1, 0]])
    input_dict = {"input": torch.from_numpy(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 2D tensor with float values
    x = np.random.randn(2, 4).astype(np.float32)
    input_dict = {"input": torch.from_numpy(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 1D tensor with float values
    x = np.random.randn(5).astype(np.float32)
    input_dict = {"input": torch.from_numpy(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4: Scalar input
    x = np.float32(3.14)
    input_dict = {"input": torch.tensor(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: 2D tensor with negative values
    x = np.array([[-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    input_dict = {"input": torch.from_numpy(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.corrcoef"] = corrcoef_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.corrcoef' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.corrcoef'.")

check_valid('torch.corrcoef', generated_inputs['torch.corrcoef'], lib="torch")
