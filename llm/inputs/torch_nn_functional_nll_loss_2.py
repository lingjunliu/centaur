
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def nll_loss_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[-0.5, -1.0, -1.5], [-2.0, -2.5, -3.0]], dtype=np.float32)
    target_tensor = np.array([0, 2], dtype=np.int64)
    log_target_tensor = np.array([], dtype=np.int64)
    weight_tensor = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    size_average_bool = True
    ignore_index_int = -100
    reduce_bool = True
    reduction_str = 'mean'

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "log_target": log_target_tensor,
        "weight": weight_tensor,
        "size_average": size_average_bool,
        "ignore_index": ignore_index_int,
        "reduce": reduce_bool,
        "reduction": reduction_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[-0.5, -1.0, -1.5]], dtype=np.float32)
    target_tensor = np.array([0], dtype=np.int64)
    log_target_tensor = np.array([], dtype=np.int64)
    weight_tensor = None
    size_average_bool = False
    ignore_index_int = -100
    reduce_bool = False
    reduction_str = 'none'

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "log_target": log_target_tensor,
        "weight": weight_tensor,
        "size_average": size_average_bool,
        "ignore_index": ignore_index_int,
        "reduce": reduce_bool,
        "reduction": reduction_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[-0.5, -1.0, -1.5], [-2.0, -2.5, -3.0]], dtype=np.float32)
    target_tensor = np.array([0, 2], dtype=np.int64)
    log_target_tensor = np.array([], dtype=np.int64)
    weight_tensor = np.array(1.0, dtype=np.float32)
    size_average_bool = True
    ignore_index_int = 0
    reduce_bool = True
    reduction_str = 'sum'

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "log_target": log_target_tensor,
        "weight": weight_tensor,
        "size_average": size_average_bool,
        "ignore_index": ignore_index_int,
        "reduce": reduce_bool,
        "reduction": reduction_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_tensor = np.array([[-0.5, -1.0, -1.5]], dtype=np.float32)
    target_tensor = np.array([0], dtype=np.int64)
    log_target_tensor = np.array([], dtype=np.int64)
    weight_tensor = None
    size_average_bool = False
    ignore_index_int = 0
    reduce_bool = True
    reduction_str = 'mean'

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "log_target": log_target_tensor,
        "weight": weight_tensor,
        "size_average": size_average_bool,
        "ignore_index": ignore_index_int,
        "reduce": reduce_bool,
        "reduction": reduction_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.nll_loss_2"] = nll_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.nll_loss_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.nll_loss_2'.")

check_valid('torch.nn.functional.nll_loss', generated_inputs['torch.nn.functional.nll_loss_2'], lib="torch", suffix=2)
