
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def get_rng_state_inputs():
    list_of_inputs = []

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.get_rng_state"] = get_rng_state_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.get_rng_state' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.get_rng_state'.")

check_valid('torch.get_rng_state', generated_inputs['torch.get_rng_state'], lib="torch")
