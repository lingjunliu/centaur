
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def hsplit_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D tensor, split into 2
    t = torch.arange(16.0).reshape(4, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensor, split with specific indices
    t = torch.arange(16.0).reshape(4, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [1, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D tensor, split into 2 (divisible size)
    t = torch.arange(24.0).reshape(2, 4, 3).numpy()
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D tensor, split into 3 (acts like tensor_split dim=0)
    t = torch.arange(9.0).numpy()
    input_dict = {"input": t, "indices_or_sections": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Complex tensor, split into 2
    t = torch.complex(torch.arange(4.0), torch.arange(4.0)).reshape(1, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Integer tensor
    t = torch.arange(16).reshape(4, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.hsplit_3"] = hsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.hsplit_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hsplit_3'.")

check_valid('torch.hsplit', generated_inputs['torch.hsplit_3'], lib="torch")
