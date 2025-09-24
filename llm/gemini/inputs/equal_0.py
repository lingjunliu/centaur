
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_equal_inputs():
    list_of_inputs = []

    # Case 1: Equal integer tensors
    input1 = torch.tensor([1, 2, 3]).numpy()
    input2 = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Unequal integer tensors
    input1 = torch.tensor([1, 2, 3]).numpy()
    input2 = torch.tensor([1, 2, 4]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Equal float tensors
    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input2 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: Unequal float tensors with different shapes
    input1 = torch.tensor([1.0, 2.0]).numpy()
    input2 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Equal multi-dimensional tensors
    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    input2 = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Unequal multi-dimensional tensors
    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    input2 = torch.tensor([[1, 2], [3, 5]]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Tensors with negative values
    input1 = torch.tensor([-1, -2, -3]).numpy()
    input2 = torch.tensor([-1, -2, -3]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Empty Tensors
    input1 = torch.tensor([]).numpy()
    input2 = torch.tensor([]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.equal"] = torch_equal_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.equal'.")

check_valid('torch.equal', generated_inputs['torch.equal'], lib="torch")
