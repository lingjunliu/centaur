
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def miopen_batch_norm_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(2, 3, 4, 5).cuda().cpu().numpy()
    weight = torch.randn(3).cuda().cpu().numpy()
    bias = torch.randn(3).cuda().cpu().numpy()
    running_mean = torch.randn(3).cuda().cpu().numpy()
    running_var = torch.rand(3).cuda().cpu().numpy()
    training = True
    exponential_average_factor = 0.1
    eps = 1e-5

    input_dict = {
        "input": input_tensor,
        "weight": weight,
        "bias": bias,
        "running_mean": running_mean,
        "running_var": running_var,
        "training": training,
        "exponential_average_factor": exponential_average_factor,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(1, 5, 10).cuda().cpu().numpy()
    weight = torch.randn(5).cuda().cpu().numpy()
    bias = torch.randn(5).cuda().cpu().numpy()
    running_mean = torch.randn(5).cuda().cpu().numpy()
    running_var = torch.rand(5).cuda().cpu().numpy()
    training = False
    exponential_average_factor = 0.5
    eps = 1e-8

    input_dict = {
        "input": input_tensor,
        "weight": weight,
        "bias": bias,
        "running_mean": running_mean,
        "running_var": running_var,
        "training": training,
        "exponential_average_factor": exponential_average_factor,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(4, 2, 2, 2).cuda().cpu().numpy()
    weight = torch.randn(2).cuda().cpu().numpy()
    bias = torch.randn(2).cuda().cpu().numpy()
    running_mean = torch.randn(2).cuda().cpu().numpy()
    running_var = torch.rand(2).cuda().cpu().numpy()
    training = True
    exponential_average_factor = 0.9
    eps = 1e-4

    input_dict = {
        "input": input_tensor,
        "weight": weight,
        "bias": bias,
        "running_mean": running_mean,
        "running_var": running_var,
        "training": training,
        "exponential_average_factor": exponential_average_factor,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_tensor = torch.randn(3, 7).cuda().cpu().numpy()
    weight = torch.randn(7).cuda().cpu().numpy()
    bias = torch.randn(7).cuda().cpu().numpy()
    running_mean = torch.randn(7).cuda().cpu().numpy()
    running_var = torch.rand(7).cuda().cpu().numpy()
    training = False
    exponential_average_factor = 0.2
    eps = 1e-6

    input_dict = {
        "input": input_tensor,
        "weight": weight,
        "bias": bias,
        "running_mean": running_mean,
        "running_var": running_var,
        "training": training,
        "exponential_average_factor": exponential_average_factor,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_tensor = torch.randn(1, 4, 3).cuda().cpu().numpy()
    weight = torch.randn(4).cuda().cpu().numpy()
    bias = torch.randn(4).cuda().cpu().numpy()
    running_mean = torch.randn(4).cuda().cpu().numpy()
    running_var = torch.rand(4).cuda().cpu().numpy()
    training = True
    exponential_average_factor = 0.75
    eps = 1e-7

    input_dict = {
        "input": input_tensor,
        "weight": weight,
        "bias": bias,
        "running_mean": running_mean,
        "running_var": running_var,
        "training": training,
        "exponential_average_factor": exponential_average_factor,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.miopen_batch_norm"] = miopen_batch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.miopen_batch_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.miopen_batch_norm'.")

check_valid('torch.miopen_batch_norm', generated_inputs['torch.miopen_batch_norm'], lib="torch", suffix=0)
