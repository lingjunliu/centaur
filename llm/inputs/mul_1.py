
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_mul_inputs():
    list_of_inputs = []

    # Case 1: Basic multiplication with float tensors
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Multiplication with a scalar (number)
    input2 = torch.randn(5).numpy()
    other2 = 2.5
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Multiplication with integer tensors
    input3 = torch.randint(0, 10, (2, 2)).numpy()
    other3 = torch.randint(0, 5, (2, 2)).numpy()
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Broadcasting - input (4,1), other (1,4)
    input4 = torch.randn(4, 1).numpy()
    other4 = torch.randn(1, 4).numpy()
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Broadcasting - input (5,3,2), other (3,2)
    input5 = torch.randn(5, 3, 2).numpy()
    other5 = torch.randn(3, 2).numpy()
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.mul_1"] = torch_mul_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.mul_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mul_1'.")

check_valid('torch.mul', generated_inputs['torch.mul_1'], lib="torch")
