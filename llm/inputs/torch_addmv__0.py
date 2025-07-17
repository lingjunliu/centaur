
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def addmv_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mat = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    beta = 1.0
    alpha = 1.0
    input_dict = {"input": input_tensor, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([1.0, 2.0], dtype=np.float32)
    mat = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    vec = np.array([1.0, 2.0], dtype=np.float32)
    beta = 0.0
    alpha = 2.0
    input_dict = {"input": input_tensor, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.addmv_"] = addmv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.addmv_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addmv_'.")

check_valid('torch.addmv_', generated_inputs['torch.addmv_'], lib="torch", suffix=0)
