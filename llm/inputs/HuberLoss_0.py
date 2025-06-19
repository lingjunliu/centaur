
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def HuberLoss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with default parameters
    input = torch.randn(3, 5).numpy()
    target = torch.randn(3, 5).numpy()
    input_dict = {
        "reduction": 'mean',
        "delta": 1.0,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different reduction method ('sum')
    input = torch.randn(2, 4, 6).numpy()
    target = torch.randn(2, 4, 6).numpy()
    input_dict = {
        "reduction": 'sum',
        "delta": 1.0,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: No reduction
    input = torch.randn(1, 7).numpy()
    target = torch.randn(1, 7).numpy()
    input_dict = {
        "reduction": 'none',
        "delta": 1.0,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different delta value
    input = torch.randn(4, 4).numpy()
    target = torch.randn(4, 4).numpy()
    input_dict = {
        "reduction": 'mean',
        "delta": 0.5,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative values in input and target
    input = torch.randn(2, 3) * -1.0
    target = torch.randn(2, 3) * -1.0
    input = input.numpy()
    target = target.numpy()
    input_dict = {
        "reduction": 'mean',
        "delta": 1.0,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.HuberLoss"] = HuberLoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.HuberLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.HuberLoss'.")

check_valid('torch.nn.HuberLoss', generated_inputs['torch.nn.HuberLoss'], lib="torch")
