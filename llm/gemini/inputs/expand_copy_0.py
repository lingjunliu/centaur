
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def expand_copy_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor
    self = torch.randn(3).numpy()
    size = (2, 3)
    input_dict = {"self": self, "size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensor
    self = torch.randn(2, 1).numpy()
    size = (2, 3)
    input_dict = {"self": self, "size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D tensor
    self = torch.randn(1, 2, 3).numpy()
    size = (4, 2, 3)
    input_dict = {"self": self, "size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Integer tensor
    self = torch.randint(0, 10, (2, 1)).numpy()
    size = (2, 5)
    input_dict = {"self": self, "size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Float tensor, larger size
    self = torch.randn(1, 5, 1, 7).numpy()
    size = (2, 5, 4, 7)
    input_dict = {"self": self, "size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Tensor of zeros
    self = torch.zeros(1, 3).numpy()
    size = (4, 3)
    input_dict = {"self": self, "size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Complex tensor
    self = torch.complex(torch.randn(2, 1), torch.randn(2, 1)).numpy()
    size = (2, 3)
    input_dict = {"self": self, "size": size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.expand_copy"] = expand_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.expand_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.expand_copy'.")

check_valid('torch.expand_copy', generated_inputs['torch.expand_copy'], lib="torch")
