
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_vsplit_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor, split into 2
    input_tensor = torch.arange(16.0).reshape(4, 4).numpy()
    indices_or_sections = (2,)
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, split into 3 unequal parts
    input_tensor = torch.arange(20.0).reshape(5, 4).numpy()
    indices_or_sections = (2,4)
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, split into 2
    input_tensor = torch.arange(24.0).reshape(2, 3, 4).numpy()
    indices_or_sections = (1,)
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D tensor, split into 4 equal parts
    input_tensor = torch.arange(32.0).reshape(8, 4).numpy()
    indices_or_sections = (2, 4, 6)
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float tensor
    input_tensor = torch.randn(4, 4).numpy()
    indices_or_sections = (2,)
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Int tensor
    input_tensor = torch.randint(0, 10, (4, 4)).numpy()
    indices_or_sections = (2,)
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Complex tensor (converted to float64 as complex64/128 are not supported)
    input_tensor = torch.randn(4, 4, dtype=torch.complex64).numpy().astype(np.float64)
    indices_or_sections = (2,)
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.vsplit_3"] = torch_vsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.vsplit_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vsplit_3'.")

check_valid('torch.vsplit', generated_inputs['torch.vsplit_3'], lib="torch")
