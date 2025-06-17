
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def miopen_batch_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32 data
    input1 = torch.randn(2, 3, 4, 5).numpy()
    weight1 = torch.randn(3).numpy()
    bias1 = torch.randn(3).numpy()
    running_mean1 = torch.randn(3).numpy()
    running_var1 = np.abs(torch.rand(3).numpy())  # Ensure running_var is non-negative
    
    input_dict1 = {
        "input": input1,
        "weight": weight1,
        "bias": bias1,
        "running_mean": running_mean1,
        "running_var": running_var1,
        "training": True,
        "exponential_average_factor": 0.1,
        "eps": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D input, float64 data
    input2 = torch.randn(2, 3, 4).double().numpy()
    weight2 = torch.randn(3).double().numpy()
    bias2 = torch.randn(3).double().numpy()
    running_mean2 = torch.randn(3).double().numpy()
    running_var2 = np.abs(torch.rand(3).double().numpy())  # Ensure running_var is non-negative

    input_dict2 = {
        "input": input2,
        "weight": weight2,
        "bias": bias2,
        "running_mean": running_mean2,
        "running_var": running_var2,
        "training": False,
        "exponential_average_factor": 0.2,
        "eps": 1e-8
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 5D input, small batch size
    input3 = torch.randn(1, 2, 3, 4, 5).numpy()
    weight3 = torch.randn(2).numpy()
    bias3 = torch.randn(2).numpy()
    running_mean3 = torch.randn(2).numpy()
    running_var3 = np.abs(torch.rand(2).numpy())  # Ensure running_var is non-negative
    
    input_dict3 = {
        "input": input3,
        "weight": weight3,
        "bias": bias3,
        "running_mean": running_mean3,
        "running_var": running_var3,
        "training": True,
        "exponential_average_factor": 0.5,
        "eps": 1e-3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D input
    input4 = torch.randn(5, 4).numpy()
    weight4 = torch.randn(4).numpy()
    bias4 = torch.randn(4).numpy()
    running_mean4 = torch.randn(4).numpy()
    running_var4 = np.abs(torch.rand(4).numpy())  # Ensure running_var is non-negative
    
    input_dict4 = {
        "input": input4,
        "weight": weight4,
        "bias": bias4,
        "running_mean": running_mean4,
        "running_var": running_var4,
        "training": False,
        "exponential_average_factor": 0.9,
        "eps": 1e-6
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different exponential average factor
    input5 = torch.randn(3, 5, 5).numpy()
    weight5 = torch.randn(5).numpy()
    bias5 = torch.randn(5).numpy()
    running_mean5 = torch.randn(5).numpy()
    running_var5 = np.abs(torch.rand(5).numpy())  # Ensure running_var is non-negative

    input_dict5 = {
        "input": input5,
        "weight": weight5,
        "bias": bias5,
        "running_mean": running_mean5,
        "running_var": running_var5,
        "training": True,
        "exponential_average_factor": 0.0,
        "eps": 1e-4
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.miopen_batch_norm"] = miopen_batch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.miopen_batch_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.miopen_batch_norm'.")

check_valid('torch.miopen_batch_norm', generated_inputs['torch.miopen_batch_norm'], lib="torch")
