
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def InstanceNorm1d_inputs():
    list_of_inputs = []

    num_features = 100
    input1 = torch.randn(20, num_features, 40).numpy()
    input_dict1 = {
        "num_features": num_features,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "dtype": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    num_features = 50
    input2 = torch.randn(5, num_features, 20).numpy()
    input_dict2 = {
        "num_features": num_features,
        "eps": 1e-04,
        "momentum": 0.2,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    num_features = 25
    input3 = torch.randn(10, num_features, 10).numpy()
    input_dict3 = {
        "num_features": num_features,
        "eps": 1e-06,
        "momentum": 0.05,
        "affine": False,
        "track_running_stats": True,
        "dtype": None,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    num_features = 64
    input4 = torch.randn(1, num_features, 32).numpy()
    input_dict4 = {
        "num_features": num_features,
        "eps": 1e-07,
        "momentum": 0.9,
        "affine": True,
        "track_running_stats": False,
        "dtype": None,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    num_features = 128
    input5 = torch.randn(32, num_features, 64).numpy()
    input_dict5 = {
        "num_features": num_features,
        "eps": 1e-03,
        "momentum": 0.3,
        "affine": False,
        "track_running_stats": True,
        "dtype": None,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.InstanceNorm1d"] = InstanceNorm1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.InstanceNorm1d', generated_inputs['torch.nn.InstanceNorm1d'], lib="torch")
