
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def deg2rad_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, single element
    input1 = np.array(180.0)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.deg2rad"] = deg2rad_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.deg2rad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.deg2rad'.")

check_valid('torch.deg2rad', generated_inputs['torch.deg2rad'], lib="torch")
