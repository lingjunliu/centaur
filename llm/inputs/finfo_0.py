
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def finfo_inputs():
    list_of_inputs = []

    input1 = {'dtype': torch.float16}
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {'dtype': torch.float32}
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {'dtype': torch.float64}
    list_of_inputs.append(copy.deepcopy(input3))

    input4 = {'dtype': torch.bfloat16}
    list_of_inputs.append(copy.deepcopy(input4))

    return list_of_inputs

generated_inputs["torch.finfo"] = finfo_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.finfo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.finfo'.")

check_valid('torch.finfo', generated_inputs['torch.finfo'], lib="torch")
