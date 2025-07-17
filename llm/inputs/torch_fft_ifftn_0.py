
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_fft_ifftn_inputs():
    list_of_inputs = []

    # Input 1: Basic example with complex input
    input1 = torch.randn(8, 8, dtype=torch.complex64).numpy()
    input_dict1 = {"input": input1, "s": (8,8), "dim": (0,1), "norm": "backward", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Specifying s
    input2 = torch.randn(16, 16, dtype=torch.complex64).numpy()
    s2 = (8, 8)
    input_dict2 = {"input": input2, "s": s2, "dim": (0,1), "norm": "backward", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.ifftn"] = torch_fft_ifftn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.ifftn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.ifftn'.")

check_valid('torch.fft.ifftn', generated_inputs['torch.fft.ifftn'], lib="torch", suffix=0)
