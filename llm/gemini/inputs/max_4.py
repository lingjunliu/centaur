
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_max_4_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive floats
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative floats
    input2 = torch.randn(2, 5) * -1.0
    other2 = torch.randn(2, 5) * -1.0
    input2 = input2.numpy()
    other2 = other2.numpy()
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Integer tensors
    input3 = torch.randint(0, 10, (4, 3)).numpy()
    other3 = torch.randint(0, 10, (4, 3)).numpy()
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different shapes, broadcasting
    input4 = torch.randn(2, 1).numpy()
    other4 = torch.randn(2, 3).numpy()
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: 3D Tensor
    input5 = torch.randn(2, 3, 4).numpy()
    other5 = torch.randn(2, 3, 4).numpy()
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.max_4"] = torch_max_4_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.max_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.max_4'.")

check_valid('torch.max', generated_inputs['torch.max_4'], lib="torch")
