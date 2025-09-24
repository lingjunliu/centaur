
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def l1_loss_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors, reduction='mean'
    input_dict = {
        "input": torch.randn(3, 5).numpy(),
        "target": torch.randn(3, 5).numpy(),
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors, reduction='sum'
    input_dict = {
        "input": torch.randint(-5, 5, (2, 4), dtype=torch.int32).numpy(),
        "target": torch.randint(-5, 5, (2, 4), dtype=torch.int32).numpy(),
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different shapes, reduction='none'
    input_dict = {
        "input": torch.randn(1, 6, 7, 8).numpy(),
        "target": torch.randn(1, 6, 7, 8).numpy(),
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Scalar tensors, reduction='mean'
    input_dict = {
        "input": torch.randn(1).numpy(),
        "target": torch.randn(1).numpy(),
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative values, reduction='sum'
    input_dict = {
        "input": torch.randn(2, 3) * -1.0,
        "target": torch.randn(2, 3) * -1.0,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["torch.nn.functional.l1_loss"] = l1_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.l1_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.l1_loss'.")

check_valid('torch.nn.functional.l1_loss', generated_inputs['torch.nn.functional.l1_loss'], lib="torch")
