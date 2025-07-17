
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def normal_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive mean
    mean = torch.arange(1., 6.).float()
    std = 2.0
    out = torch.empty(5).float()
    input_dict = {"mean": mean.numpy(), "std": std, "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.normal_3"] = normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.normal_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.normal_3'.")

check_valid('torch.normal', generated_inputs['torch.normal_3'], lib="torch", suffix=3)
