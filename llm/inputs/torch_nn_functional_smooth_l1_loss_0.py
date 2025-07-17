
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def smooth_l1_loss_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target_tensor = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    reduction_type = "mean"
    delta_value = 1.0

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_type,
        "delta": delta_value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target_tensor = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    reduction_type = "sum"
    delta_value = 0.5

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_type,
        "delta": delta_value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    target_tensor = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    reduction_type = "none"
    delta_value = 0.2

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_type,
        "delta": delta_value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    target_tensor = np.array([-1.5, -2.5, -3.5], dtype=np.float32)
    reduction_type = "mean"
    delta_value = 1.5

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_type,
        "delta": delta_value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    target_tensor = np.array([[-1.1, 2.2], [-3.3, 4.4]], dtype=np.float32)
    reduction_type = "sum"
    delta_value = 0.75

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_type,
        "delta": delta_value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    target_tensor = np.array([1.1, 2.1, 3.1], dtype=np.float64)
    reduction_type = "none"
    delta_value = 0.1

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_type,
        "delta": delta_value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([1.0], dtype=np.float32)
    target_tensor = np.array([1.0], dtype=np.float32)
    reduction_type = "mean"
    delta_value = 0.0

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_type,
        "delta": delta_value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    target_tensor = np.array([[-1.1, -2.1, -3.1], [-4.1, -5.1, -6.1]], dtype=np.float32)
    reduction_type = "sum"
    delta_value = 2.0

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_type,
        "delta": delta_value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([0.0], dtype=np.float32)
    target_tensor = np.array([0.0], dtype=np.float32)
    reduction_type = "none"
    delta_value = 1.0

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_type,
        "delta": delta_value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    target_tensor = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    reduction_type = "mean"
    delta_value = 0.3

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_type,
        "delta": delta_value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    target_tensor = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float32)
    reduction_type = "mean"
    delta_value = 0.3

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_type,
        "delta": delta_value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.smooth_l1_loss"] = smooth_l1_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.smooth_l1_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.smooth_l1_loss'.")

check_valid('torch.nn.functional.smooth_l1_loss', generated_inputs['torch.nn.functional.smooth_l1_loss'], lib="torch", suffix=0)
