
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def promote_types_inputs():
    list_of_inputs = []

    input_dict = {
        "type1": torch.float32,
        "type2": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "type1": torch.int32,
        "type2": torch.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "type1": torch.uint8,
        "type2": torch.int16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "type1": torch.complex64,
        "type2": torch.complex128
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "type1": torch.float16,
        "type2": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "type1": torch.bool,
        "type2": torch.int8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.promote_types"] = promote_types_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.promote_types' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.promote_types'.")

check_valid('torch.promote_types', generated_inputs['torch.promote_types'], lib="torch")
