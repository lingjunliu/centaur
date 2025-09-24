
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def affine_grid_inputs():
    list_of_inputs = []

    # Input 1
    theta = np.array([[[0.5, 0, 0], [0, 0.5, 0]]], dtype=np.float32)
    size = (1, 1, 10, 10)
    align_corners = True
    input_dict = {"theta": theta, "size": size, "align_corners": align_corners}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    theta = np.array([[[1.0, 0, 0], [0, 1.0, 0]]], dtype=np.float32)
    size = (1, 3, 20, 30)
    align_corners = False
    input_dict = {"theta": theta, "size": size, "align_corners": align_corners}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    theta = np.array([[[0.7, 0.2, 0.1], [-0.2, 0.7, 0.3]]], dtype=np.float32)
    size = (1, 1, 40, 50)
    align_corners = True
    input_dict = {"theta": theta, "size": size, "align_corners": align_corners}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    theta = np.array([[[1.2, -0.1, 0.0], [0.1, 0.9, 0.0]]], dtype=np.float32)
    size = (1, 1, 256, 256)
    align_corners = False
    input_dict = {"theta": theta, "size": size, "align_corners": align_corners}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    theta = np.array([[[0.9, 0, 0], [0, 0.9, 0]]], dtype=np.float32)
    size = (1, 1, 64, 64)
    align_corners = True
    input_dict = {"theta": theta, "size": size, "align_corners": align_corners}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    theta = np.array([[[0.6, 0, 0.2], [0, 0.6, -0.1]]], dtype=np.float32)
    size = (1, 1, 128, 128)
    align_corners = False
    input_dict = {"theta": theta, "size": size, "align_corners": align_corners}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    theta = np.array([[[0.8, 0.1, 0.1], [-0.1, 0.8, 0.2]]], dtype=np.float32)
    size = (1, 1, 40, 50)
    align_corners = False
    input_dict = {"theta": theta, "size": size, "align_corners": align_corners}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    theta = np.array([[[1.0, 0, 0.2], [0, 1.0, -0.3]]], dtype=np.float32)
    size = (1, 1, 32, 32)
    align_corners = True
    input_dict = {"theta": theta, "size": size, "align_corners": align_corners}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: multiple batches
    theta = np.array([[[1.0, 0, 0.2], [0, 1.0, -0.3]], [[0.5, 0, 0], [0, 0.5, 0]]], dtype=np.float32)
    size = (2, 1, 32, 32)
    align_corners = True
    input_dict = {"theta": theta, "size": size, "align_corners": align_corners}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: multiple batches, different size
    theta = np.array([[[1.0, 0, 0.2], [0, 1.0, -0.3]], [[0.5, 0, 0], [0, 0.5, 0]]], dtype=np.float32)
    size = (2, 1, 64, 64)
    align_corners = False
    input_dict = {"theta": theta, "size": size, "align_corners": align_corners}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.affine_grid"] = affine_grid_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.affine_grid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.affine_grid'.")

check_valid('torch.nn.functional.affine_grid', generated_inputs['torch.nn.functional.affine_grid'], lib="torch", suffix=0)
