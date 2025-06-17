
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def atan2_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = torch.randn(4).numpy()
    input2 = torch.randn(4).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors
    input1 = torch.randint(-5, 5, (3,)).numpy()
    input2 = torch.randint(1, 10, (3,)).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different shapes (broadcastable)
    input1 = torch.randn(2, 3).numpy()
    input2 = torch.randn(3).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Negative values
    input1 = torch.randn(2, 2).numpy() * -1
    input2 = torch.randn(2, 2).numpy() * -1
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Zero values
    input1 = torch.zeros(5).numpy()
    input2 = torch.ones(5).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.atan2"] = atan2_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.atan2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atan2'.")

check_valid('torch.atan2', generated_inputs['torch.atan2'], lib="torch")
