
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def cudnn_affine_grid_generator_inputs():
    list_of_inputs = []

    # Input 1
    theta = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]).cuda().cpu().numpy()
    N = np.int64(1)
    C = np.int64(1)
    H = np.int64(10)
    W = np.int64(20)

    input_dict = {
        "theta": theta,
        "N": N,
        "C": C,
        "H": H,
        "W": W
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).cuda().cpu().numpy()
    N = np.int64(2)
    C = np.int64(3)
    H = np.int64(32)
    W = np.int64(32)

    input_dict = {
        "theta": theta,
        "N": N,
        "C": C,
        "H": H,
        "W": W
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.cudnn_affine_grid_generator"] = cudnn_affine_grid_generator_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cudnn_affine_grid_generator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cudnn_affine_grid_generator'.")

check_valid('torch.cudnn_affine_grid_generator', generated_inputs['torch.cudnn_affine_grid_generator'], lib="torch", suffix=0)
