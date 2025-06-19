
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sync_batchnorm_inputs():
    list_of_inputs = []

    # Input 1
    num_features = np.int32(100)
    eps = np.float32(1e-05)
    momentum = np.float32(0.1)
    affine = np.bool_(True)
    track_running_stats = np.bool_(True)
    process_group = []
    dtype = None

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
    num_features = np.int32(50)
    eps = np.float32(1e-04)
    momentum = np.float32(0.2)
    affine = np.bool_(False)
    track_running_stats = np.bool_(False)
    process_group = []
    dtype = None

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
generated_inputs["torch.nn.SyncBatchNorm"] = sync_batchnorm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.SyncBatchNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.SyncBatchNorm'.")

check_valid('torch.nn.SyncBatchNorm', generated_inputs['torch.nn.SyncBatchNorm'], lib="torch", suffix=0)
