
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy
import torch.distributed as dist

def sync_batchnorm_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "num_features": 10,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "process_group": [],
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "num_features": 20,
        "eps": 1e-4,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "process_group": [],
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "num_features": 5,
        "eps": 1e-6,
        "momentum": 0.05,
        "affine": True,
        "track_running_stats": True,
        "process_group": [],
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "num_features": 15,
        "eps": 1e-3,
        "momentum": 0.3,
        "affine": False,
        "track_running_stats": False,
        "process_group": [],
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "num_features": 8,
        "eps": 1e-7,
        "momentum": 0.08,
        "affine": True,
        "track_running_stats": True,
        "process_group": [],
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "num_features": 32,
        "eps": 1e-2,
        "momentum": 0.4,
        "affine": False,
        "track_running_stats": False,
        "process_group": [],
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "num_features": 64,
        "eps": 1e-8,
        "momentum": 0.01,
        "affine": True,
        "track_running_stats": True,
        "process_group": [],
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "num_features": 128,
        "eps": 1e-1,
        "momentum": 0.5,
        "affine": False,
        "track_running_stats": False,
        "process_group": [],
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "num_features": 256,
        "eps": 1e-9,
        "momentum": 0.001,
        "affine": True,
        "track_running_stats": True,
        "process_group": [],
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "num_features": 512,
        "eps": 1e-10,
        "momentum": 0.6,
        "affine": False,
        "track_running_stats": False,
        "process_group": [],
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.SyncBatchNorm"] = sync_batchnorm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.SyncBatchNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.SyncBatchNorm'.")

check_valid('torch.nn.SyncBatchNorm', generated_inputs['torch.nn.SyncBatchNorm'], lib="torch", suffix=0)
