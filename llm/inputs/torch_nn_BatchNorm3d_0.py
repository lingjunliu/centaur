
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def batchnorm3d_inputs():
    list_of_inputs = []

    # Input 1
    num_features = 10
    eps = 1e-5
    momentum = 0.1
    affine = True
    track_running_stats = True
    input_tensor = torch.randn(2, 10, 4, 4, 4).numpy()
    input_dict = {"num_features": num_features, "eps": eps, "momentum": momentum, "affine": affine, "track_running_stats": track_running_stats, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    num_features = 5
    eps = 1e-4
    momentum = 0.2
    affine = False
    track_running_stats = False
    input_tensor = torch.randn(1, 5, 8, 8, 8).numpy()
    input_dict = {"num_features": num_features, "eps": eps, "momentum": momentum, "affine": affine, "track_running_stats": track_running_stats, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    num_features = 20
    eps = 1e-6
    momentum = 0.05
    affine = True
    track_running_stats = True
    input_tensor = torch.randn(4, 20, 16, 16, 16).numpy()
    input_dict = {"num_features": num_features, "eps": eps, "momentum": momentum, "affine": affine, "track_running_stats": track_running_stats, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    num_features = 3
    eps = 1e-3
    momentum = 0.9
    affine = False
    track_running_stats = False
    input_tensor = torch.randn(8, 3, 32, 32, 32).numpy()
    input_dict = {"num_features": num_features, "eps": eps, "momentum": momentum, "affine": affine, "track_running_stats": track_running_stats, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    num_features = 64
    eps = 1e-7
    momentum = 0.01
    affine = True
    track_running_stats = True
    input_tensor = torch.randn(3, 64, 2, 2, 2).numpy()
    input_dict = {"num_features": num_features, "eps": eps, "momentum": momentum, "affine": affine, "track_running_stats": track_running_stats, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    num_features = 128
    eps = 1e-2
    momentum = 0.5
    affine = False
    track_running_stats = False
    input_tensor = torch.randn(5, 128, 10, 10, 10).numpy()
    input_dict = {"num_features": num_features, "eps": eps, "momentum": momentum, "affine": affine, "track_running_stats": track_running_stats, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (modified to avoid ValueError)
    num_features = 1
    eps = 1e-8
    momentum = 0.99
    affine = True
    track_running_stats = True
    input_tensor = torch.randn(2, 1, 2, 2, 2).numpy()  # Increased dimensions
    input_dict = {"num_features": num_features, "eps": eps, "momentum": momentum, "affine": affine, "track_running_stats": track_running_stats, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    num_features = 256
    eps = 1e-1
    momentum = 0.001
    affine = False
    track_running_stats = False
    input_tensor = torch.randn(16, 256, 5, 5, 5).numpy()
    input_dict = {"num_features": num_features, "eps": eps, "momentum": momentum, "affine": affine, "track_running_stats": track_running_stats, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    num_features = 4
    eps = 0.0
    momentum = 0.0
    affine = True
    track_running_stats = True
    input_tensor = torch.randn(1, 4, 5, 5, 5).numpy()
    input_dict = {"num_features": num_features, "eps": eps, "momentum": momentum, "affine": affine, "track_running_stats": track_running_stats, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    num_features = 8
    eps = 0.5
    momentum = 0.7
    affine = False
    track_running_stats = False
    input_tensor = torch.randn(2, 8, 7, 7, 7).numpy()
    input_dict = {"num_features": num_features, "eps": eps, "momentum": momentum, "affine": affine, "track_running_stats": track_running_stats, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.BatchNorm3d"] = batchnorm3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.BatchNorm3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BatchNorm3d'.")

check_valid('torch.nn.BatchNorm3d', generated_inputs['torch.nn.BatchNorm3d'], lib="torch", suffix=0)
