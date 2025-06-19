
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import torch.nn as nn
import copy
import numpy as np

def DataParallel_inputs():
    list_of_inputs = []

    # Input 1
    module = nn.Linear(10, 5)
    device_ids = [0]
    output_device = 0
    dim = 0
    input_dict = {
        "module": module,
        "device_ids": device_ids,
        "output_device": output_device,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy({
        "module": "nn.Module",
        "device_ids": np.array(device_ids),
        "output_device": np.array(output_device),
        "dim": np.array(dim)
    }))

    # Input 2
    module = nn.Conv2d(3, 16, kernel_size=3)
    device_ids = [0]
    output_device = 0
    dim = 0
    input_dict = {
        "module": module,
        "device_ids": device_ids,
        "output_device": output_device,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy({
        "module": "nn.Module",
        "device_ids": np.array(device_ids),
        "output_device": np.array(output_device),
        "dim": np.array(dim)
    }))

    # Input 3
    module = nn.Sequential(
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
    )
    device_ids = [0]
    output_device = 0
    dim = 0
    input_dict = {
        "module": module,
        "device_ids": device_ids,
        "output_device": output_device,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy({
        "module": "nn.Module",
        "device_ids": np.array(device_ids),
        "output_device": np.array(output_device),
        "dim": np.array(dim)
    }))

    # Input 4
    module = nn.LSTM(10, 20, batch_first=True)
    device_ids = [0]
    output_device = 0
    dim = 0 # Different dim
    input_dict = {
        "module": module,
        "device_ids": device_ids,
        "output_device": output_device,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy({
        "module": "nn.Module",
        "device_ids": np.array(device_ids),
        "output_device": np.array(output_device),
        "dim": np.array(dim)
    }))

    # Input 5: Module with batch norm
    module = nn.Sequential(
        nn.Linear(10, 5),
        nn.BatchNorm1d(5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    device_ids = [0]
    output_device = 0
    dim = 0
    input_dict = {
        "module": module,
        "device_ids": device_ids,
        "output_device": output_device,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy({
        "module": "nn.Module",
        "device_ids": np.array(device_ids),
        "output_device": np.array(output_device),
        "dim": np.array(dim)
    }))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.DataParallel"] = DataParallel_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.DataParallel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.DataParallel'.")

check_valid('torch.nn.DataParallel', generated_inputs['torch.nn.DataParallel'], lib="torch", suffix=0)
