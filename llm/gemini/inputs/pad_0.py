
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def pad_inputs():
    list_of_inputs = []

    # Test case 1: 4D tensor, constant padding
    input_tensor = torch.randn(3, 3, 4, 2).numpy()
    pad = (1, 1, 2, 2)
    mode = 'constant'
    value = 0.0
    input_dict = {"input": input_tensor, "pad": pad, "mode": mode, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: 3D tensor, replicate padding
    input_tensor = torch.randn(3, 4, 5).numpy()
    pad = (1, 1)
    mode = 'replicate'
    value = None
    input_dict = {"input": input_tensor, "pad": pad, "mode": mode, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 3: 2D tensor, reflect padding
    input_tensor = torch.randn(4, 6).numpy()
    pad = (1, 1)
    mode = 'reflect'
    value = None
    input_dict = {"input": input_tensor, "pad": pad, "mode": mode, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    # Test case 5: 4D tensor, constant padding with a different value
    input_tensor = torch.randn(2, 4, 5, 3).numpy()
    pad = (2, 0, 1, 1)
    mode = 'constant'
    value = -1.5
    input_dict = {"input": input_tensor, "pad": pad, "mode": mode, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.pad"] = pad_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.pad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pad'.")

check_valid('torch.nn.functional.pad', generated_inputs['torch.nn.functional.pad'], lib="torch")
