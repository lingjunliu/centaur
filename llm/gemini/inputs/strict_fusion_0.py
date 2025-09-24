
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def strict_fusion_inputs():
    list_of_inputs = []
    
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    
    return list_of_inputs

generated_inputs["torch.jit.strict_fusion"] = strict_fusion_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.strict_fusion' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.strict_fusion'.")

check_valid('torch.jit.strict_fusion', generated_inputs['torch.jit.strict_fusion'], lib="torch")
