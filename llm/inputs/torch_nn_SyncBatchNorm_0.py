
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy
import torch.distributed as dist

def syncbatchnorm_inputs():
    list_of_inputs = []

    # Input 1
    num_features = np.int32(10)
    eps = np.float32(1e-5)
    momentum = np.float32(0.1)
    affine = np.bool_(True)
    track_running_stats = np.bool_(True)
    process_group = None if not (dist.is_available() and dist.is_initialized()) else [1,2]
    dtype = torch.float32

    input_dict = {
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats,
        "process_group": process_group,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    num_features = np.int32(5)
    eps = np.float32(1e-4)
    momentum = np.float32(0.2)
    affine = np.bool_(False)
    track_running_stats = np.bool_(False)
    process_group = None if not (dist.is_available() and dist.is_initialized()) else [3,4]
    dtype = torch.float64

    input_dict = {
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats,
        "process_group": process_group,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    num_features = np.int32(20)
    eps = np.float32(1e-6)
    momentum = np.float32(0.05)
    affine = np.bool_(True)
    track_running_stats = np.bool_(True)
    process_group = None if not (dist.is_available() and dist.is_initialized()) else [5,6]
    dtype = torch.float16

    input_dict = {
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats,
        "process_group": process_group,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    num_features = np.int32(1)
    eps = np.float32(1e-7)
    momentum = np.float32(0.9)
    affine = np.bool_(False)
    track_running_stats = np.bool_(False)
    process_group = None if not (dist.is_available() and dist.is_initialized()) else [7,8]
    dtype = torch.float32
    
    input_dict = {
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats,
        "process_group": process_group,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    num_features = np.int32(100)
    eps = np.float32(1e-3)
    momentum = np.float32(0.01)
    affine = np.bool_(True)
    track_running_stats = np.bool_(True)
    process_group = None if not (dist.is_available() and dist.is_initialized()) else [9,10]
    dtype = torch.float64

    input_dict = {
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats,
        "process_group": process_group,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    num_features = np.int32(2)
    eps = np.float32(1e-8)
    momentum = np.float32(0.5)
    affine = np.bool_(False)
    track_running_stats = np.bool_(False)
    process_group = None if not (dist.is_available() and dist.is_initialized()) else [11,12]
    dtype = torch.float16

    input_dict = {
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats,
        "process_group": process_group,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    num_features = np.int32(64)
    eps = np.float32(1.0)
    momentum = np.float32(0.3)
    affine = np.bool_(True)
    track_running_stats = np.bool_(False)
    process_group = None if not (dist.is_available() and dist.is_initialized()) else [13,14]
    dtype = torch.float32

    input_dict = {
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats,
        "process_group": process_group,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    num_features = np.int32(32)
    eps = np.float32(0.5)
    momentum = np.float32(0.7)
    affine = np.bool_(False)
    track_running_stats = np.bool_(True)
    process_group = None if not (dist.is_available() and dist.is_initialized()) else [15,16]
    dtype = torch.float64
    
    input_dict = {
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats,
        "process_group": process_group,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    num_features = np.int32(8)
    eps = np.float32(0.0)
    momentum = np.float32(0.0)
    affine = np.bool_(True)
    track_running_stats = np.bool_(True)
    process_group = None if not (dist.is_available() and dist.is_initialized()) else [17,18]
    dtype = torch.float16
    
    input_dict = {
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats,
        "process_group": process_group,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    num_features = np.int32(4)
    eps = np.float32(0.1)
    momentum = np.float32(1.0)
    affine = np.bool_(False)
    track_running_stats = np.bool_(False)
    process_group = None if not (dist.is_available() and dist.is_initialized()) else [19,20]
    dtype = torch.float32

    input_dict = {
        "num_features": num_features,
        "eps": eps,
        "momentum": momentum,
        "affine": affine,
        "track_running_stats": track_running_stats,
        "process_group": process_group,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.SyncBatchNorm"] = syncbatchnorm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.SyncBatchNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.SyncBatchNorm'.")

check_valid('torch.nn.SyncBatchNorm', generated_inputs['torch.nn.SyncBatchNorm'], lib="torch", suffix=0)
