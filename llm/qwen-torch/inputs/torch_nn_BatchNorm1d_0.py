
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def batchnorm1d_inputs():
    list_of_inputs = []

    
    # Input 1, valid
    input = torch.randn(20, 100).numpy()
    num_features = 100
    eps = 1e-5
    momentum = 0.1
    affine = True
    track_running_stats = True
    
    input_dict = {
        "input": input,
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.randn(10, 50).numpy()
    num_features = 50
    eps = 1e-4
    momentum = 0.2
    affine = False
    track_running_stats = True
    
    input_dict = {
        "input": input,
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(15, 200).numpy()
    num_features = 200
    eps = 1e-6
    momentum = 0.05
    affine = True
    track_running_stats = False
    
    input_dict = {
        "input": input,
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(5, 25).numpy()
    num_features = 25
    eps = 1e-5
    momentum = 0.1
    affine = False
    track_running_stats = False
    
    input_dict = {
        "input": input,
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(30, 10).numpy()
    num_features = 10
    eps = 1e-5
    momentum = 0.01
    affine = True
    track_running_stats = True
    
    input_dict = {
        "input": input,
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(12, 75).numpy()
    num_features = 75
    eps = 1e-4
    momentum = 0.3
    affine = False
    track_running_stats = True
    
    input_dict = {
        "input": input,
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(8, 300).numpy()
    num_features = 300
    eps = 1e-6
    momentum = 0.15
    affine = True
    track_running_stats = False
    
    input_dict = {
        "input": input,
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(25, 150).numpy()
    num_features = 150
    eps = 1e-5
    momentum = 0.05
    affine = False
    track_running_stats = True
    
    input_dict = {
        "input": input,
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(100, 50).numpy()
    num_features = 50
    eps = 1e-4
    momentum = 0.2
    affine = True
    track_running_stats = False
    
    input_dict = {
        "input": input,
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(50, 20).numpy()
    num_features = 20
    eps = 1e-5
    momentum = 0.1
    affine = False
    track_running_stats = True
    
    input_dict = {
        "input": input,
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11, valid
    input = torch.randn(35, 100).numpy()
    num_features = 100
    eps = 1e-6
    momentum = 0.01
    affine = True
    track_running_stats = True
    
    input_dict = {
        "input": input,
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12, valid
    input = torch.randn(15, 30).numpy()
    num_features = 30
    eps = 1e-5
    momentum = 0.1
    affine = False
    track_running_stats = False
    
    input_dict = {
        "input": input,
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

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
