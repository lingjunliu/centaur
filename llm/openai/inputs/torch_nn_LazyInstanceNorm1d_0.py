
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def lazy_instance_norm1d_inputs():
    list_of_inputs = []

    # All inputs use consistent dtype and channel size to avoid dtype/num_features mismatches.
    # Use C=4 and L>=2 to avoid spatial size errors in training mode.
    # 1
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": np.random.randn(2, 4, 8).astype(np.float32)
    }))

    # 2
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-4,
        "momentum": 0.0,
        "affine": False,
        "track_running_stats": False,
        "dtype": None,
        "input": np.random.randn(3, 4, 5).astype(np.float32)
    }))

    # 3
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-6,
        "momentum": 0.9,
        "affine": True,
        "track_running_stats": False,
        "dtype": None,
        "input": (np.random.rand(1, 4, 6).astype(np.float32) - 0.5) * 3.0
    }))

    # 4
    list_of_inputs.append(copy.deepcopy({
        "eps": 5e-5,
        "momentum": 0.2,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": np.full((4, 4, 3), 2.0, dtype=np.float32)
    }))

    # 5
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-3,
        "momentum": 0.5,
        "affine": False,
        "track_running_stats": True,
        "dtype": None,
        "input": (np.random.randn(5, 4, 7).astype(np.float32) * 2.5)
    }))

    # 6
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-7,
        "momentum": 0.3,
        "affine": True,
        "track_running_stats": False,
        "dtype": None,
        "input": np.linspace(-1.0, 1.0, num=2*4*4, dtype=np.float32).reshape(2, 4, 4)
    }))

    # 7
    list_of_inputs.append(copy.deepcopy({
        "eps": 2e-5,
        "momentum": 0.7,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": np.random.randn(1, 4, 10).astype(np.float32)
    }))

    # 8
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-2,
        "momentum": 0.01,
        "affine": False,
        "track_running_stats": False,
        "dtype": None,
        "input": (np.random.randn(6, 4, 9).astype(np.float32) + 5.0)
    }))

    # 9
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-8,
        "momentum": 0.6,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": np.random.randn(8, 4, 2).astype(np.float32)
    }))

    # 10
    list_of_inputs.append(copy.deepcopy({
        "eps": 3e-5,
        "momentum": 0.4,
        "affine": False,
        "track_running_stats": True,
        "dtype": None,
        "input": (np.random.rand(2, 4, 12).astype(np.float32) - 0.5) * 10.0
    }))

    return list_of_inputs

generated_inputs["torch.nn.LazyInstanceNorm1d"] = lazy_instance_norm1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LazyInstanceNorm1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LazyInstanceNorm1d'.")


check_valid('torch.nn.LazyInstanceNorm1d', generated_inputs['torch.nn.LazyInstanceNorm1d'], lib="torch", suffix=0)
