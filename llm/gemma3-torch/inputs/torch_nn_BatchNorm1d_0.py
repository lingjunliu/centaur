
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def batchnorm1d_inputs():
    list_of_inputs = []
    
    input1 = {'num_features': 10, 'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True, 'dtype': np.float32}
    list_of_inputs.append(copy.deepcopy(input1))
    
    return list_of_inputs

generated_inputs["torch.nn.BatchNorm1d"] = batchnorm1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.BatchNorm1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BatchNorm1d'.")


check_valid('torch.nn.BatchNorm1d', generated_inputs['torch.nn.BatchNorm1d'], lib="torch", suffix=0)
