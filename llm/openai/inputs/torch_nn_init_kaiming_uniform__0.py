
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def kaiming_uniform_inputs():
    list_of_inputs = []

    tensor = np.zeros((8, 16), dtype=np.float32)
    input_dict = {"tensor": tensor, "a": 0.0, "mode": "fan_in", "nonlinearity": "relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.ones((32, 64), dtype=np.float64)
    input_dict = {"tensor": tensor, "a": 0.01, "mode": "fan_in", "nonlinearity": "leaky_relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.random.randn(64, 3, 7, 7).astype(np.float32)
    input_dict = {"tensor": tensor, "a": 0.0, "mode": "fan_out", "nonlinearity": "relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.random.uniform(-1, 1, (32, 16, 5)).astype(np.float16)
    input_dict = {"tensor": tensor, "a": 0.2, "mode": "fan_in", "nonlinearity": "leaky_relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.random.randn(16, 8, 3, 3, 3).astype(np.float32)
    input_dict = {"tensor": tensor, "a": 0.0, "mode": "fan_in", "nonlinearity": "relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = -np.abs(np.random.randn(10, 10).astype(np.float32))
    input_dict = {"tensor": tensor, "a": -0.2, "mode": "fan_out", "nonlinearity": "leaky_relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = (np.random.randn(32, 32, 1, 1) * 0.1).astype(np.float64)
    input_dict = {"tensor": tensor, "a": 1.5, "mode": "fan_out", "nonlinearity": "leaky_relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = (np.random.rand(128, 64, 3) - 0.5).astype(np.float32)
    input_dict = {"tensor": tensor, "a": 0.0, "mode": "fan_out", "nonlinearity": "relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.random.randn(16, 16, 5, 5).astype(np.float16)
    input_dict = {"tensor": tensor, "a": 0.5, "mode": "fan_in", "nonlinearity": "leaky_relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.random.randn(1, 1024).astype(np.float64)
    input_dict = {"tensor": tensor, "a": 2.0, "mode": "fan_in", "nonlinearity": "relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.random.randn(8, 8, 2, 2, 2).astype(np.float32)
    input_dict = {"tensor": tensor, "a": 0.01, "mode": "fan_out", "nonlinearity": "leaky_relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = (np.random.randn(256, 256) * 0.01).astype(np.float32)
    input_dict = {"tensor": tensor, "a": 0.0, "mode": "fan_out", "nonlinearity": "relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.init.kaiming_uniform_"] = kaiming_uniform_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.kaiming_uniform_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.kaiming_uniform_'.")


check_valid('torch.nn.init.kaiming_uniform_', generated_inputs['torch.nn.init.kaiming_uniform_'], lib="torch", suffix=0)
