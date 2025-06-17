
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def broadcast_tensors_inputs():
    list_of_inputs = []

    # Case 1: Basic broadcasting
    tensors = [torch.tensor([1, 2, 3]).numpy(), torch.tensor([[4], [5], [6]]).numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Scalars and vectors
    tensors = [torch.tensor(1).numpy(), torch.tensor([2, 3, 4]).numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Broadcasting with higher dimensions
    tensors = [torch.tensor([[[1, 2]], [[3, 4]]]).numpy(), torch.tensor([[5, 6]]).numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: Mismatched dtypes that can be promoted
    tensors = [torch.tensor([1, 2], dtype=torch.int32).numpy(), torch.tensor([1.0, 2.0], dtype=torch.float64).numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Broadcasting with different sized dimensions
    tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy(), torch.tensor([7, 8, 9]).numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.broadcast_tensors"] = broadcast_tensors_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.broadcast_tensors' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.broadcast_tensors'.")

check_valid('torch.broadcast_tensors', generated_inputs['torch.broadcast_tensors'], lib="torch")
