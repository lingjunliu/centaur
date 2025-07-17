
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def generate_upsample_inputs():
    list_of_inputs = []

    # Input 1: 4D input, scale_factor, nearest
    input1 = np.arange(1, 17, dtype=np.float32).reshape((1, 1, 4, 4))
    scale_factor1 = (2.0, 2.0)
    size1 = None
    mode1 = 'nearest'
    align_corners1 = None
    recompute_scale_factor1 = None
    input_dict1 = {"size": size1, "scale_factor": scale_factor1, "mode": mode1, "align_corners": align_corners1, "recompute_scale_factor": recompute_scale_factor1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Upsample_3"] = generate_upsample_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Upsample_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Upsample_3'.")

check_valid('torch.nn.Upsample', generated_inputs['torch.nn.Upsample_3'], lib="torch", suffix=3)
