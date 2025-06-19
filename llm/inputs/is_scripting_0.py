
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def is_scripting_inputs():
    list_of_inputs = []

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.jit.is_scripting"] = is_scripting_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.is_scripting' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.is_scripting'.")

check_valid('torch.jit.is_scripting', generated_inputs['torch.jit.is_scripting'], lib="torch")
