
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def multi_margin_loss_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    target_tensor = np.array([0, 1], dtype=np.int64)
    p_val = 1
    margin_val = 1.0
    weight_tensor = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    size_average_val = True
    reduce_val = True
    reduction_val = 'mean'

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "p": p_val,
        "margin": margin_val,
        "weight": weight_tensor,
        "size_average": size_average_val,
        "reduce": reduce_val,
        "reduction": reduction_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    target_tensor = np.array([0], dtype=np.int64)
    p_val = 2
    margin_val = 0.5
    weight_tensor = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    size_average_val = False
    reduce_val = False
    reduction_val = 'none'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "p": p_val,
        "margin": margin_val,
        "weight": weight_tensor,
        "size_average": size_average_val,
        "reduce": reduce_val,
        "reduction": reduction_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    target_tensor = np.array([2, 1], dtype=np.int64)
    p_val = 1
    margin_val = 0.8
    weight_tensor = np.array([0.3, 0.3, 0.4], dtype=np.float32)
    size_average_val = False
    reduce_val = True
    reduction_val = 'sum'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "p": p_val,
        "margin": margin_val,
        "weight": weight_tensor,
        "size_average": size_average_val,
        "reduce": reduce_val,
        "reduction": reduction_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[0.5, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    target_tensor = np.array([0, 3], dtype=np.int64)
    p_val = 2
    margin_val = 0.2
    weight_tensor = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    size_average_val = True
    reduce_val = False
    reduction_val = 'none'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "p": p_val,
        "margin": margin_val,
        "weight": weight_tensor,
        "size_average": size_average_val,
        "reduce": reduce_val,
        "reduction": reduction_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[-0.1, -0.2, -0.3]], dtype=np.float32)
    target_tensor = np.array([1], dtype=np.int64)
    p_val = 1
    margin_val = 0.9
    weight_tensor = np.array([0.6, 0.2, 0.2], dtype=np.float32)
    size_average_val = False
    reduce_val = True
    reduction_val = 'mean'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "p": p_val,
        "margin": margin_val,
        "weight": weight_tensor,
        "size_average": size_average_val,
        "reduce": reduce_val,
        "reduction": reduction_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)
    target_tensor = np.array([0, 2], dtype=np.int64)
    p_val = 2
    margin_val = 1.5
    weight_tensor = np.array([0.1, 0.1, 0.8], dtype=np.float32)
    size_average_val = True
    reduce_val = False
    reduction_val = 'none'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "p": p_val,
        "margin": margin_val,
        "weight": weight_tensor,
        "size_average": size_average_val,
        "reduce": reduce_val,
        "reduction": reduction_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    input_tensor = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
    target_tensor = np.array([3], dtype=np.int64)
    p_val = 1
    margin_val = 0.7
    weight_tensor = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    size_average_val = False
    reduce_val = True
    reduction_val = 'sum'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "p": p_val,
        "margin": margin_val,
        "weight": weight_tensor,
        "size_average": size_average_val,
        "reduce": reduce_val,
        "reduction": reduction_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0], [-3.0, -2.0, -1.0, 0.0, 1.0]], dtype=np.float32)
    target_tensor = np.array([4, 0], dtype=np.int64)
    p_val = 2
    margin_val = 1.2
    weight_tensor = np.array([0.1, 0.2, 0.2, 0.2, 0.3], dtype=np.float32)
    size_average_val = True
    reduce_val = True
    reduction_val = 'mean'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "p": p_val,
        "margin": margin_val,
        "weight": weight_tensor,
        "size_average": size_average_val,
        "reduce": reduce_val,
        "reduction": reduction_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[0.8, 0.7, 0.6, 0.5, 0.4]], dtype=np.float32)
    target_tensor = np.array([2], dtype=np.int64)
    p_val = 1
    margin_val = 0.3
    weight_tensor = np.array([0.3, 0.2, 0.2, 0.1, 0.2], dtype=np.float32)
    size_average_val = False
    reduce_val = False
    reduction_val = 'none'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "p": p_val,
        "margin": margin_val,
        "weight": weight_tensor,
        "size_average": size_average_val,
        "reduce": reduce_val,
        "reduction": reduction_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    target_tensor = np.array([1, 0, 1], dtype=np.int64)
    p_val = 2
    margin_val = 0.6
    weight_tensor = np.array([0.4, 0.6], dtype=np.float32)
    size_average_val = True
    reduce_val = True
    reduction_val = 'sum'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "p": p_val,
        "margin": margin_val,
        "weight": weight_tensor,
        "size_average": size_average_val,
        "reduce": reduce_val,
        "reduction": reduction_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    target_tensor = np.array([1], dtype=np.int64)
    p_val = 1
    margin_val = 1.0
    weight_tensor = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    size_average_val = False
    reduce_val = False
    reduction_val = 'none'

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "p": p_val,
        "margin": margin_val,
        "weight": weight_tensor,
        "size_average": size_average_val,
        "reduce": reduce_val,
        "reduction": reduction_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.multi_margin_loss"] = multi_margin_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.multi_margin_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.multi_margin_loss'.")

check_valid('torch.nn.functional.multi_margin_loss', generated_inputs['torch.nn.functional.multi_margin_loss'], lib="torch", suffix=0)
