
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def conj_physical_inputs():
    list_of_inputs = []

    # Input 1: Complex float tensor
    input1 = (torch.randn(2, 3) + 1j * torch.randn(2, 3)).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.conj_physical"] = conj_physical_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.conj_physical' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.conj_physical'.")

check_valid('torch.conj_physical', generated_inputs['torch.conj_physical'], lib="torch")
