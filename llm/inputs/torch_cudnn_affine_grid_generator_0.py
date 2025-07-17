
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def cudnn_affine_grid_generator_inputs():
    list_of_inputs = []

    # Input 1
    theta = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
    N = 1
    C = 1
    H = 10
    W = 10
    input_dict = {"theta": theta, "N": N, "C": C, "H": H, "W": W}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    theta = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    N = 2
    C = 3
    H = 20
    W = 30
    input_dict = {"theta": theta, "N": N, "C": C, "H": H, "W": W}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    theta = np.array([[0.7, 0.2, 0.1], [-0.2, 0.7, 0.2]], dtype=np.float32)
    N = 4
    C = 1
    H = 25
    W = 25
    input_dict = {"theta": theta, "N": N, "C": C, "H": H, "W": W}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    theta = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    N = 1
    C = 1
    H = 5
    W = 5
    input_dict = {"theta": theta, "N": N, "C": C, "H": H, "W": W}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    theta = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=np.float32)
    N = 2
    C = 3
    H = 15
    W = 15
    input_dict = {"theta": theta, "N": N, "C": C, "H": H, "W": W}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    theta = np.array([[0.9, -0.1, 0.2], [0.1, 0.9, -0.3]], dtype=np.float32)
    N = 3
    C = 1
    H = 32
    W = 32
    input_dict = {"theta": theta, "N": N, "C": C, "H": H, "W": W}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    theta = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    N = 1
    C = 1
    H = 1
    W = 1
    input_dict = {"theta": theta, "N": N, "C": C, "H": H, "W": W}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    theta = np.array([[0.6, 0.1, -0.2], [-0.1, 0.6, 0.3]], dtype=np.float32)
    N = 2
    C = 3
    H = 64
    W = 64
    input_dict = {"theta": theta, "N": N, "C": C, "H": H, "W": W}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    theta = np.array([[0.9, -0.4, 0.1], [0.4, 0.9, -0.1]], dtype=np.float32)
    N = 1
    C = 1
    H = 40
    W = 40
    input_dict = {"theta": theta, "N": N, "C": C, "H": H, "W": W}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    theta = np.array([[1.0, 0.0, -0.3], [0.0, 1.0, 0.4]], dtype=np.float32)
    N = 3
    C = 3
    H = 28
    W = 28
    input_dict = {"theta": theta, "N": N, "C": C, "H": H, "W": W}
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
