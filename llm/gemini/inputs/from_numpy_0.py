
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def from_numpy_inputs():
    list_of_inputs = []

    ndarray = np.array([1, 2, 3])
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(input_dict)

    ndarray = np.array([[1, 2], [3, 4]])
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.from_numpy"] = from_numpy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.from_numpy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.from_numpy'.")

check_valid('torch.from_numpy', generated_inputs['torch.from_numpy'], lib="torch", suffix=0)
