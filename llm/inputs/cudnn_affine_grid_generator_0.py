
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cudnn_affine_grid_generator_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    theta = torch.randn(1, 2, 3).cuda()
    N = 1
    C = 1
    H = 10
    W = 20
    input_dict = {
        "theta": theta,
        "N": N,
        "C": C,
        "H": H,
        "W": W
    }
    input_dict["theta"] = input_dict["theta"].cuda()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different batch size and spatial dimensions
    theta = torch.randn(2, 2, 3).cuda()
    N = 2
    C = 3
    H = 15
    W = 25
    input_dict = {
        "theta": theta,
        "N": N,
        "C": C,
        "H": H,
        "W": W
    }
    input_dict["theta"] = input_dict["theta"].cuda()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger values in theta
    theta = (torch.randn(1, 2, 3) * 10).cuda()
    N = 1
    C = 1
    H = 5
    W = 5
    input_dict = {
        "theta": theta,
        "N": N,
        "C": C,
        "H": H,
        "W": W
    }
    input_dict["theta"] = input_dict["theta"].cuda()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values in theta
    theta = (torch.randn(1, 2, 3) * -1).cuda()
    N = 1
    C = 1
    H = 7
    W = 14
    input_dict = {
        "theta": theta,
        "N": N,
        "C": C,
        "H": H,
        "W": W
    }
    input_dict["theta"] = input_dict["theta"].cuda()
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Different channel size
    theta = torch.randn(1, 2, 3).cuda()
    N = 1
    C = 5
    H = 12
    W = 18
    input_dict = {
        "theta": theta,
        "N": N,
        "C": C,
        "H": H,
        "W": W
    }
    input_dict["theta"] = input_dict["theta"].cuda()

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.cudnn_affine_grid_generator"] = cudnn_affine_grid_generator_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cudnn_affine_grid_generator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cudnn_affine_grid_generator'.")

check_valid('torch.cudnn_affine_grid_generator', generated_inputs['torch.cudnn_affine_grid_generator'], lib="torch")
