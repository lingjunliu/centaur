
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def manual_seed_inputs():
    list_of_inputs = []

    input_dict = {
        "seed": np.array(0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "seed": np.array(12345)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "seed": np.array(2147483647)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "seed": np.array(-2147483648)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "seed": np.array(1000000000)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "seed": np.array(999999999)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "seed": np.array(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.manual_seed"] = manual_seed_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.manual_seed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.manual_seed'.")

check_valid('torch.manual_seed', generated_inputs['torch.manual_seed'], lib="torch")
