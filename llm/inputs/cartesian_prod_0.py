
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cartesian_prod_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two tensors
    tensor1 = torch.tensor([1, 2], dtype=torch.int64)
    tensor2 = torch.tensor([3, 4], dtype=torch.int64)
    input_dict = {"tensors": [tensor1, tensor2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Three tensors
    tensor1 = torch.tensor([1, 2], dtype=torch.int64)
    tensor2 = torch.tensor([3], dtype=torch.int64)
    tensor3 = torch.tensor([4, 5, 6], dtype=torch.int64)
    input_dict = {"tensors": [tensor1, tensor2, tensor3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float tensors
    tensor1 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    tensor2 = torch.tensor([3.0, 4.0], dtype=torch.float32)
    input_dict = {"tensors": [tensor1, tensor2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed integer and float tensors
    tensor1 = torch.tensor([1, 2], dtype=torch.int64)
    tensor2 = torch.tensor([3.0, 4.0], dtype=torch.float32)
    input_dict = {"tensors": [tensor1, tensor2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensors with different sizes
    tensor1 = torch.tensor([1, 2, 3], dtype=torch.int64)
    tensor2 = torch.tensor([4, 5], dtype=torch.int64)
    input_dict = {"tensors": [tensor1, tensor2]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    
    tensors_list = [torch.tensor([1, 2], dtype=torch.int64), torch.tensor([3, 4], dtype=torch.int64)]
    input_dict = {"tensors": tensors_list}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.cartesian_prod"] = cartesian_prod_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cartesian_prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cartesian_prod'.")

check_valid('torch.cartesian_prod', generated_inputs['torch.cartesian_prod'], lib="torch")
