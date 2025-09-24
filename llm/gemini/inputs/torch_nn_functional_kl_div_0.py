
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def kl_div_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([0.1, 0.2, 0.7])
    target_tensor = np.array([0.3, 0.3, 0.4])
    reduction = "mean"
    log_target = False
    input_dict = {"input": input_tensor, "target": target_tensor, "reduction": reduction, "log_target": log_target}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[0.1, 0.2], [0.3, 0.4]])
    target_tensor = np.array([[0.5, 0.5], [0.2, 0.8]])
    reduction = "batchmean"
    log_target = False
    input_dict = {"input": input_tensor, "target": target_tensor, "reduction": reduction, "log_target": log_target}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[-0.1, -0.2], [-0.3, -0.4]])
    target_tensor = np.array([[0.5, 0.5], [0.2, 0.8]])
    reduction = "sum"
    log_target = True
    input_dict = {"input": input_tensor, "target": target_tensor, "reduction": reduction, "log_target": log_target}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([0.1, 0.9])
    target_tensor = np.array([0.9, 0.1])
    reduction = "none"
    log_target = False
    input_dict = {"input": input_tensor, "target": target_tensor, "reduction": reduction, "log_target": log_target}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
    target_tensor = np.array([[[0.5, 0.5], [0.2, 0.8]], [[0.1, 0.9], [0.6, 0.4]]])
    reduction = "mean"
    log_target = True
    input_dict = {"input": input_tensor, "target": target_tensor, "reduction": reduction, "log_target": log_target}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([-1.0, -2.0, -3.0])
    target_tensor = np.array([0.2, 0.3, 0.5])
    reduction = "sum"
    log_target = True
    input_dict = {"input": input_tensor, "target": target_tensor, "reduction": reduction, "log_target": log_target}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([0.5, 0.5])
    target_tensor = np.array([0.5, 0.5])
    reduction = "batchmean"
    log_target = False
    input_dict = {"input": input_tensor, "target": target_tensor, "reduction": reduction, "log_target": log_target}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[0.1]]])
    target_tensor = np.array([[[0.9]]])
    reduction = "mean"
    log_target = False
    input_dict = {"input": input_tensor, "target": target_tensor, "reduction": reduction, "log_target": log_target}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([0.0, 0.0])
    target_tensor = np.array([1.0, 0.0])
    reduction = "mean"
    log_target = True
    input_dict = {"input": input_tensor, "target": target_tensor, "reduction": reduction, "log_target": log_target}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = np.array([[-1.0, -0.5], [-0.2, -0.3]])
    target_tensor = np.array([[0.1, 0.9], [0.8, 0.2]])
    reduction = "none"
    log_target = True
    input_dict = {"input": input_tensor, "target": target_tensor, "reduction": reduction, "log_target": log_target}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.kl_div"] = kl_div_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.kl_div' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.kl_div'.")

check_valid('torch.nn.functional.kl_div', generated_inputs['torch.nn.functional.kl_div'], lib="torch", suffix=0)
