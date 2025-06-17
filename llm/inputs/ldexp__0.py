
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def ldexp_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor and integer tensor
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randint(0, 5, (3, 4)).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor and integer tensor
    input2 = torch.randn(2, 2).numpy()
    other2 = torch.randint(-3, 3, (2, 2), dtype=torch.int32).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor
    input3 = torch.randn(5).numpy()
    other3 = torch.randint(0, 3, (5,)).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar input
    input4 = torch.randn(1).numpy()
    other4 = torch.randint(-5, 5, (1,)).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Different shapes that are broadcastable
    input5 = torch.randn(2, 3).numpy()
    other5 = torch.randint(0, 4, (3,)).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.ldexp_"] = ldexp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ldexp_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ldexp_'.")

check_valid('torch.ldexp_', generated_inputs['torch.ldexp_'], lib="torch")
