
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def block_diag_inputs():
    list_of_inputs = []

    # Case 1: Basic case with two 2D tensors
    tensors = [torch.from_numpy(np.random.rand(2, 3)), torch.from_numpy(np.random.rand(4, 5))]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Three 2D tensors with different shapes
    tensors = [torch.from_numpy(np.random.rand(1, 1)), torch.from_numpy(np.random.rand(2, 2)), torch.from_numpy(np.random.rand(3, 3))]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Two 1D tensors (converted to 2D)
    tensors = [torch.from_numpy(np.random.rand(3)), torch.from_numpy(np.random.rand(4))]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Mix of 1D and 2D tensors
    tensors = [torch.from_numpy(np.random.rand(2)), torch.from_numpy(np.random.rand(3, 4)), torch.from_numpy(np.random.rand(5))]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Tensors with negative values
    tensors = [torch.from_numpy(np.random.randn(2, 2)), torch.from_numpy(np.random.randn(3, 3))]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Integer tensors
    tensors = [torch.from_numpy(np.random.randint(1, 5, size=(2, 2)).astype(np.int64)), torch.from_numpy(np.random.randint(1, 5, size=(3, 3)).astype(np.int64))]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.block_diag"] = block_diag_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.block_diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.block_diag'.")

check_valid('torch.block_diag', generated_inputs['torch.block_diag'], lib="torch")
