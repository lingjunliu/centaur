
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_ne_inputs():
    list_of_inputs = []

    # Case 1: Basic integer tensor comparison
    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    other1 = torch.tensor([[1, 1], [4, 4]]).numpy()
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Float tensor compared to a number
    input2 = torch.tensor([[1.5, 2.5], [3.5, 4.5]]).numpy()
    other2 = 3.0
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Broadcasting with scalar tensor
    input3 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    other3 = torch.tensor(2).numpy()
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Negative values
    input4 = torch.tensor([[-1, -2], [-3, -4]]).numpy()
    other4 = torch.tensor([[-2, -2], [-4, -3]]).numpy()
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Different shapes with broadcasting
    input5 = torch.tensor([[1, 2, 3]]).numpy()
    other5 = torch.tensor([1, 2, 4]).numpy()
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.ne"] = torch_ne_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ne' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ne'.")

check_valid('torch.ne', generated_inputs['torch.ne'], lib="torch")
