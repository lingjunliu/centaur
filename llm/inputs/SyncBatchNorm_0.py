
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import torch.distributed as dist
import numpy as np
import torch.nn as nn

def SyncBatchNorm_inputs():
    list_of_inputs = []

    input_dict = {
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "process_group": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_features": 50,
        "eps": 1e-04,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "process_group": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_features": 128,
        "eps": 1e-06,
        "momentum": 0.05,
        "affine": True,
        "track_running_stats": True,
        "process_group": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_features": 64,
        "eps": 1e-03,
        "momentum": None,
        "affine": False,
        "track_running_stats": True,
        "process_group": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_features": 256,
        "eps": 1e-07,
        "momentum": 0.3,
        "affine": True,
        "track_running_stats": False,
        "process_group": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.SyncBatchNorm"] = SyncBatchNorm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.SyncBatchNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.SyncBatchNorm'.")

check_valid('torch.nn.SyncBatchNorm', generated_inputs['torch.nn.SyncBatchNorm'], lib="torch")
