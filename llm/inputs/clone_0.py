
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def clone_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(3, 4, 5).numpy()
    input_dict = {"input": input_tensor, "memory_format": torch.preserve_format}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randint(-10, 10, (2, 2), dtype=torch.int32).numpy()
    input_dict = {"input": input_tensor, "memory_format": torch.preserve_format}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 5, 5, 5).numpy()
    input_dict = {"input": input_tensor, "memory_format": torch.preserve_format}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(size=(10,)).numpy()
    input_dict = {"input": input_tensor, "memory_format": torch.preserve_format}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.zeros(2, 3).numpy()
    input_dict = {"input": input_tensor, "memory_format": torch.preserve_format}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.clone"] = clone_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.clone' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clone'.")

check_valid('torch.clone', generated_inputs['torch.clone'], lib="torch")
