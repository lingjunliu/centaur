
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def is_inference_inputs():
    list_of_inputs = []

    # The torch.is_inference API does not take any inputs

    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})

    return list_of_inputs

generated_inputs["torch.is_inference"] = is_inference_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.is_inference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_inference'.")

check_valid('torch.is_inference', generated_inputs['torch.is_inference'], lib="torch")
