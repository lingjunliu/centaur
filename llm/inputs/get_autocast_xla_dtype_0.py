
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def get_autocast_xla_dtype_inputs():
    list_of_inputs = []
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.get_autocast_xla_dtype"] = get_autocast_xla_dtype_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.get_autocast_xla_dtype' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.get_autocast_xla_dtype'.")

check_valid('torch.get_autocast_xla_dtype', generated_inputs['torch.get_autocast_xla_dtype'], lib="torch", suffix=0)
