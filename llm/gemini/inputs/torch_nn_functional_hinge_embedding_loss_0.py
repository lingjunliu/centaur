
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def hinge_embedding_loss_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    target_tensor = np.array([1, -1, 1], dtype=np.int64)
    margin_value = 1.0
    reduction_type = "mean"

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "margin": margin_value,
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
    target_tensor = np.array([[1, -1], [1, -1]], dtype=np.int64)
    margin_value = 0.5
    reduction_type = "sum"

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "margin": margin_value,
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    target_tensor = np.array([1, 1, 1], dtype=np.int64)
    margin_value = 1.0
    reduction_type = "none"

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "margin": margin_value,
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([-2.0, 2.0, -2.0], dtype=np.float32)
    target_tensor = np.array([-1, -1, -1], dtype=np.int64)
    margin_value = 2.0
    reduction_type = "mean"

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "margin": margin_value,
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[-1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    target_tensor = np.array([[-1, 1], [-1, 1]], dtype=np.int64)
    margin_value = 1.5
    reduction_type = "sum"

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "margin": margin_value,
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([0.5, -0.5, 0.5], dtype=np.float32)
    target_tensor = np.array([1, -1, 1], dtype=np.int64)
    margin_value = 0.7
    reduction_type = "none"

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "margin": margin_value,
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([1.5, -1.5], dtype=np.float32)
    target_tensor = np.array([1, -1], dtype=np.int64)
    margin_value = 1.2
    reduction_type = "mean"
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "margin": margin_value,
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[-0.5, 0.5], [-1.0, 1.0]], dtype=np.float32)
    target_tensor = np.array([[-1, 1], [-1, 1]], dtype=np.int64)
    margin_value = 0.9
    reduction_type = "sum"

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "margin": margin_value,
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([0.2, -0.2, 0.2], dtype=np.float32)
    target_tensor = np.array([1, -1, 1], dtype=np.int64)
    margin_value = 0.3
    reduction_type = "none"

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "margin": margin_value,
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[-1.2, 1.2], [0.8, -0.8]], dtype=np.float32)
    target_tensor = np.array([[-1, 1], [1, -1]], dtype=np.int64)
    margin_value = 1.8
    reduction_type = "mean"

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "margin": margin_value,
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.hinge_embedding_loss"] = hinge_embedding_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.hinge_embedding_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hinge_embedding_loss'.")

check_valid('torch.nn.functional.hinge_embedding_loss', generated_inputs['torch.nn.functional.hinge_embedding_loss'], lib="torch", suffix=0)
