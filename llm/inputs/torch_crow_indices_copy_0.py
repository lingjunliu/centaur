
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def crow_indices_copy_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    indices = np.array([0, 2, 4, 1, 3, 5], dtype=np.int64)
    row_offsets = np.array([0, 3, 6], dtype=np.int64)
    input_dict = {"indices": indices, "row_offsets": row_offsets}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.crow_indices_copy"] = crow_indices_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.crow_indices_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.crow_indices_copy'.")

check_valid('torch.crow_indices_copy', generated_inputs['torch.crow_indices_copy'], lib="torch", suffix=0)
