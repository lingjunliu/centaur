
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def flatten_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D tensor
    input_tensor = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D tensor with specific start and end dims
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "start_dim": 1, "end_dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 4D tensor with negative end_dim
    input_tensor = torch.randn(1, 2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "start_dim": 1, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Integer tensor
    input_tensor = torch.randint(0, 10, (3, 5)).numpy()
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Float tensor
    input_tensor = torch.randn(2, 2, 2).double().numpy()
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: 1D tensor
    input_tensor = torch.arange(5).numpy()
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Start dim equals to end dim.
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "start_dim": 1, "end_dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Different shape
    input_tensor = torch.randn(5, 1, 2).numpy()
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 9: Zero dim tensor to test for the specific edge case.
    input_tensor = torch.tensor(5).numpy()
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: Using int64
    input_tensor = torch.randint(0, 10, (3, 5), dtype=torch.int64).numpy()
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.flatten"] = flatten_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.flatten' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.flatten'.")

check_valid('torch.flatten', generated_inputs['torch.flatten'], lib="torch")
