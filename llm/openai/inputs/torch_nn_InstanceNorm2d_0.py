
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def instance_norm2d_inputs():
    list_of_inputs = []

    inp = torch.randn(2, 3, 8, 8).numpy()
    input_dict = {
        "num_features": 3,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 16, 16).numpy()
    input_dict = {
        "num_features": 3,
        "eps": 1e-3,
        "momentum": 0.0,
        "affine": True,
        "track_running_stats": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 1, 2).numpy()
    input_dict = {
        "num_features": 1,
        "eps": 1e-6,
        "momentum": 1.0,
        "affine": True,
        "track_running_stats": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(4, 10, 7, 5).numpy()
    input_dict = {
        "num_features": 10,
        "eps": 1e-4,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(5, 2, 3).numpy()
    input_dict = {
        "num_features": 5,
        "eps": 1e-2,
        "momentum": 0.5,
        "affine": True,
        "track_running_stats": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(8, 64, 2, 2).numpy()
    input_dict = {
        "num_features": 64,
        "eps": 1e-5,
        "momentum": 0.3,
        "affine": True,
        "track_running_stats": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(7, 9, 11).numpy()
    input_dict = {
        "num_features": 7,
        "eps": 1e-7,
        "momentum": 0.15,
        "affine": False,
        "track_running_stats": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 2, 32, 24).numpy()
    input_dict = {
        "num_features": 2,
        "eps": 1e-8,
        "momentum": 0.75,
        "affine": True,
        "track_running_stats": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(5, 3, 1, 10).numpy()
    input_dict = {
        "num_features": 3,
        "eps": 1e-5,
        "momentum": 0.05,
        "affine": False,
        "track_running_stats": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(4, 3, 5).numpy()
    input_dict = {
        "num_features": 4,
        "eps": 1e-5,
        "momentum": 0.6,
        "affine": True,
        "track_running_stats": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 32, 13, 13).numpy()
    input_dict = {
        "num_features": 32,
        "eps": 5e-5,
        "momentum": 0.0,
        "affine": False,
        "track_running_stats": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 6, 1, 3).numpy()
    input_dict = {
        "num_features": 6,
        "eps": 1e-4,
        "momentum": 0.25,
        "affine": True,
        "track_running_stats": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.InstanceNorm2d"] = instance_norm2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.InstanceNorm2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.InstanceNorm2d'.")


check_valid('torch.nn.InstanceNorm2d', generated_inputs['torch.nn.InstanceNorm2d'], lib="torch", suffix=0)
