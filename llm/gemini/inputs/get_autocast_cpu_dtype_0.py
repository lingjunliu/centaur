
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def get_autocast_cpu_dtype_inputs():
    list_of_inputs = []
    return list_of_inputs

generated_inputs["torch.get_autocast_cpu_dtype"] = get_autocast_cpu_dtype_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.get_autocast_cpu_dtype' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.get_autocast_cpu_dtype'.")

check_valid('torch.get_autocast_cpu_dtype', generated_inputs['torch.get_autocast_cpu_dtype'], lib="torch")
