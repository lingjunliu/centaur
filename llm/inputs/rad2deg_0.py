
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def rad2deg_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensor
    input1 = np.array([[np.pi/2, np.pi], [0, -np.pi/4]], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.rad2deg"] = rad2deg_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.rad2deg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rad2deg'.")

check_valid('torch.rad2deg', generated_inputs['torch.rad2deg'], lib="torch")
