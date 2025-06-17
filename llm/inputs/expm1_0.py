
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def expm1_inputs():
    list_of_inputs = []

    input1 = np.array([0, 1, -1, 0.5, -0.5], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.expm1"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.expm1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.expm1'.")

check_valid('torch.expm1', generated_inputs['torch.expm1'], lib="torch")
