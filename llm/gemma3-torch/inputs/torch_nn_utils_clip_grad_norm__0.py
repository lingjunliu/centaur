
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clip_grad_norm__inputs():
    list_of_inputs = []

    parameters1 = [np.array([1.0, 2.0, 3.0])]
    max_norm1 = 2.0
    norm_type1 = 2
    eps1 = 1e-8
    input_dict1 = {
        "parameters": parameters1,
        "max_norm": max_norm1,
        "norm_type": norm_type1,
        "eps": eps1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    parameters2 = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    max_norm2 = 3.0
    norm_type2 = 1
    eps2 = 1e-6
    input_dict2 = {
        "parameters": parameters2,
        "max_norm": max_norm2,
        "norm_type": norm_type2,
        "eps": eps2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.nn.utils.clip_grad_norm_"] = clip_grad_norm__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.clip_grad_norm_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.clip_grad_norm_'.")


check_valid('torch.nn.utils.clip_grad_norm_', generated_inputs['torch.nn.utils.clip_grad_norm_'], lib="torch", suffix=0)
