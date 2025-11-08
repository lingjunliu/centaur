
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def set_flush_denormal_inputs():
    list_of_inputs = []
    
    list_of_inputs.append(copy.deepcopy({"mode": True}))
    list_of_inputs.append(copy.deepcopy({"mode": False}))
    list_of_inputs.append(copy.deepcopy({"mode": bool(np.uint8(1))}))
    list_of_inputs.append(copy.deepcopy({"mode": bool(np.uint8(0))}))
    list_of_inputs.append(copy.deepcopy({"mode": bool(np.int64(-1))}))
    list_of_inputs.append(copy.deepcopy({"mode": bool(np.int64(0))}))
    list_of_inputs.append(copy.deepcopy({"mode": bool(np.float32(0.0))}))
    list_of_inputs.append(copy.deepcopy({"mode": bool(np.float32(0.0001))}))
    list_of_inputs.append(copy.deepcopy({"mode": bool(np.float64(np.nan))}))
    list_of_inputs.append(copy.deepcopy({"mode": bool(np.float64(0.0))}))
    list_of_inputs.append(copy.deepcopy({"mode": bool(np.bool_(True))}))
    list_of_inputs.append(copy.deepcopy({"mode": bool(np.bool_(False))}))
    
    return list_of_inputs

generated_inputs["torch.set_flush_denormal"] = set_flush_denormal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.set_flush_denormal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_flush_denormal'.")


check_valid('torch.set_flush_denormal', generated_inputs['torch.set_flush_denormal'], lib="torch", suffix=0)
