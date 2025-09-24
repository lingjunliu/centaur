
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_min_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Integer tensors
    input2 = torch.randint(0, 10, (2, 5)).numpy()
    other2 = torch.randint(0, 10, (2, 5)).numpy()
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Negative values
    input3 = torch.randint(-10, 0, (4, 3)).numpy()
    other3 = torch.randint(-5, 5, (4, 3)).numpy()
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Different shapes (broadcasting)
    input4 = torch.randn(5, 1).numpy()
    other4 = torch.randn(1, 5).numpy()
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: 1D tensors
    input5 = torch.randn(7).numpy()
    other5 = torch.randn(7).numpy()
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: 3D tensors
    input6 = torch.randn(2, 3, 4).numpy()
    other6 = torch.randn(2, 3, 4).numpy()
    input_dict6 = {"input": input6, "other": other6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Case 7: input > other
    input7 = torch.randint(5, 10, (2, 5)).numpy()
    other7 = torch.randint(0, 5, (2, 5)).numpy()
    input_dict7 = {"input": input7, "other": other7, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.min_3"] = torch_min_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.min_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.min_3'.")

check_valid('torch.min', generated_inputs['torch.min_3'], lib="torch")
