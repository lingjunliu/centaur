
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def margin_ranking_loss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float tensors, margin = 0, reduction = 'mean'
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input2 = np.array([2.0, 2.0, 4.0], dtype=np.float32)
    target = np.array([1, -1, 1], dtype=np.int32)
    margin = 0.0
    reduction = 'mean'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different sized tensors, margin = 0.5, reduction = 'sum'
    input1 = np.array([1.0, 2.0], dtype=np.float32)
    input2 = np.array([2.0, 2.0], dtype=np.float32)
    target = np.array([1, -1], dtype=np.int32)
    margin = 0.5
    reduction = 'sum'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative values, margin = 1.0, reduction = 'none'
    input1 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    input2 = np.array([0.0, -1.0, 2.0], dtype=np.float32)
    target = np.array([1, -1, 1], dtype=np.int32)
    margin = 1.0
    reduction = 'none'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Multi-dimensional arrays
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input2 = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    target = np.array([[1, -1], [1, -1]], dtype=np.int32)
    margin = 0.2
    reduction = 'mean'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different margin, reduction = 'sum'
    input1 = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target = np.array([-1, 1, -1], dtype=np.int32)
    margin = 0.7
    reduction = 'sum'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.margin_ranking_loss"] = margin_ranking_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.margin_ranking_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.margin_ranking_loss'.")

check_valid('torch.nn.functional.margin_ranking_loss', generated_inputs['torch.nn.functional.margin_ranking_loss'], lib="torch")
