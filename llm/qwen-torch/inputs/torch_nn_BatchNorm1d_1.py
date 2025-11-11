
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def batchnorm1d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with learnable parameters
    input = torch.randn(20, 100).numpy()
    input_dict = {
        "num_features": 100,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Without learnable parameters
    input = torch.randn(10, 50).numpy()
    input_dict = {
        "num_features": 50,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - With different momentum value
    input = torch.randn(15, 200).numpy()
    input_dict = {
        "num_features": 200,
        "eps": 1e-5,
        "momentum": 0.01,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With custom epsilon value
    input = torch.randn(5, 75).numpy()
    input_dict = {
        "num_features": 75,
        "eps": 1e-3,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With different batch size and features
    input = torch.randn(2, 30).numpy()
    input_dict = {
        "num_features": 30,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Negative values in input tensor
    input = torch.randn(10, 10).numpy()
    input_dict = {
        "num_features": 10,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Different dimensions (3D)
    input = torch.randn(2, 5, 10).numpy()
    input_dict = {
        "num_features": 5,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - No track running stats
    input = torch.randn(5, 20).numpy()
    input_dict = {
        "num_features": 20,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Different momentum value (None)
    input = torch.randn(3, 15).numpy()
    input_dict = {
        "num_features": 15,
        "eps": 1e-5,
        "momentum": None,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - With large number of features
    input = torch.randn(2, 1000).numpy()
    input_dict = {
        "num_features": 1000,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.BatchNorm1d_1"] = batchnorm1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.BatchNorm1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BatchNorm1d_1'.")


check_valid('torch.nn.BatchNorm1d', generated_inputs['torch.nn.BatchNorm1d_1'], lib="torch", suffix=1)
