
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def block_diag_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two 2D tensors
    tensors1 = [torch.tensor(np.random.randn(2, 3)), torch.tensor(np.random.randn(3, 4))]
    input_dict1 = {"tensors": tensors1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different shapes and data types
    tensors2 = [torch.tensor(np.random.randint(0, 10, (1, 1), dtype=np.int32)), torch.tensor(np.random.randn(2, 2))]
    input_dict2 = {"tensors": tensors2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: List of 1D tensors
    tensors3 = [torch.tensor(np.random.randn(3)), torch.tensor(np.random.randn(2))]
    input_dict3 = {"tensors": tensors3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Two 3D tensors
    tensors4 = [torch.tensor(np.random.randn(1, 2, 3)), torch.tensor(np.random.randn(2, 1, 2))]
    input_dict4 = {"tensors": tensors4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Single 2D tensor
    tensors5 = [torch.tensor(np.random.randn(3, 3))]
    input_dict5 = {"tensors": tensors5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.block_diag"] = block_diag_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.block_diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.block_diag'.")

check_valid('torch.block_diag', generated_inputs['torch.block_diag'], lib="torch")
