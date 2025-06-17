
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def torch_jit_ignore_inputs():
    list_of_inputs = []

    input_dict = {
        "drop": True
    }
    list_of_inputs.append(input_dict)


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.jit.ignore"] = torch_jit_ignore_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.ignore' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.ignore'.")

check_valid('torch.jit.ignore', generated_inputs['torch.jit.ignore'], lib="torch")
