
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def angle_inputs():
    list_of_inputs = []

    # Example 1: Complex tensor
    input1 = torch.tensor([-1 + 1j]).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: Real tensor (positive and negative)
    input2 = torch.tensor([-1.0]).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.angle"] = angle_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.angle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.angle'.")

check_valid('torch.angle', generated_inputs['torch.angle'], lib="torch")
