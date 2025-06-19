
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def dsplit_inputs():
    list_of_inputs = []

    # Case 1: Basic 3D tensor, splitting into 2
    input_tensor = torch.arange(24.0).reshape(2, 3, 4).numpy()
    indices_or_sections = (2,)
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D tensor, splitting at specific indices
    input_tensor = torch.arange(27.0).reshape(3, 3, 3).numpy()
    indices_or_sections = (1, 2)
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 4D tensor, splitting into 3
    input_tensor = torch.arange(48.0).reshape(2, 2, 3, 4).numpy()
    indices_or_sections = (1, 2)
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Float tensor
    input_tensor = torch.randn(2, 2, 4).numpy()
    indices_or_sections = (2,)
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Larger tensor
    input_tensor = torch.arange(120.0).reshape(2, 5, 12).numpy()
    indices_or_sections = (4, 8)
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Uneven split
    input_tensor = torch.arange(30.0).reshape(2, 3, 5).numpy()
    indices_or_sections = (2,4)
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.dsplit_3"] = dsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.dsplit_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dsplit_3'.")

check_valid('torch.dsplit', generated_inputs['torch.dsplit_3'], lib="torch")
