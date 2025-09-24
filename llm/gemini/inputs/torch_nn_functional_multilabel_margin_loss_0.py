
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def multilabel_margin_loss_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[0.1, 0.2, 0.4, 0.8]], dtype=np.float32)
    target_tensor = np.array([[0, 3, -1, -1]], dtype=np.int64)
    size_average_bool = True
    reduce_bool = True
    reduction_str = 'mean'
    input_dict = {"input": input_tensor, "target": target_tensor, "size_average": size_average_bool, "reduce": reduce_bool, "reduction": reduction_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[0.1, 0.2, 0.4, 0.8], [0.9, 0.7, 0.5, 0.3]], dtype=np.float32)
    target_tensor = np.array([[0, 3, -1, -1], [2, 1, -1, -1]], dtype=np.int64)
    size_average_bool = False
    reduce_bool = False
    reduction_str = 'none'
    input_dict = {"input": input_tensor, "target": target_tensor, "size_average": size_average_bool, "reduce": reduce_bool, "reduction": reduction_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[-0.1, 0.2, -0.4, 0.8]], dtype=np.float32)
    target_tensor = np.array([[0, 3, -1, -1]], dtype=np.int64)
    size_average_bool = True
    reduce_bool = False
    reduction_str = 'sum'
    input_dict = {"input": input_tensor, "target": target_tensor, "size_average": size_average_bool, "reduce": reduce_bool, "reduction": reduction_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[0.1, -0.2, 0.4, -0.8], [0.9, -0.7, 0.5, -0.3]], dtype=np.float32)
    target_tensor = np.array([[0, 3, -1, -1], [2, 1, -1, -1]], dtype=np.int64)
    size_average_bool = False
    reduce_bool = True
    reduction_str = 'mean'
    input_dict = {"input": input_tensor, "target": target_tensor, "size_average": size_average_bool, "reduce": reduce_bool, "reduction": reduction_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    target_tensor = np.array([[0, 1, -1, -1]], dtype=np.int64)
    size_average_bool = False
    reduce_bool = True
    reduction_str = 'sum'
    input_dict = {"input": input_tensor, "target": target_tensor, "size_average": size_average_bool, "reduce": reduce_bool, "reduction": reduction_str}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_tensor = np.array([[-0.5, -0.6, -0.7, -0.8]], dtype=np.float32)
    target_tensor = np.array([[0, 1, -1, -1]], dtype=np.int64)
    size_average_bool = True
    reduce_bool = False
    reduction_str = 'none'
    input_dict = {"input": input_tensor, "target": target_tensor, "size_average": size_average_bool, "reduce": reduce_bool, "reduction": reduction_str}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_tensor = np.array([[0.5, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    target_tensor = np.array([[0, 1, -1, -1], [2, 3, -1, -1]], dtype=np.int64)
    size_average_bool = False
    reduce_bool = True
    reduction_str = 'none'
    input_dict = {"input": input_tensor, "target": target_tensor, "size_average": size_average_bool, "reduce": reduce_bool, "reduction": reduction_str}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_tensor = np.array([[0.2, 0.3, 0.5, 0.7]], dtype=np.float32)
    target_tensor = np.array([[0, 2, -1, -1]], dtype=np.int64)
    size_average_bool = True
    reduce_bool = True
    reduction_str = 'mean'
    input_dict = {"input": input_tensor, "target": target_tensor, "size_average": size_average_bool, "reduce": reduce_bool, "reduction": reduction_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[-0.2, -0.3, -0.5, -0.7], [0.1, 0.4, 0.6, 0.9]], dtype=np.float32)
    target_tensor = np.array([[1, 3, -1, -1], [0, 2, -1, -1]], dtype=np.int64)
    size_average_bool = False
    reduce_bool = False
    reduction_str = 'sum'
    input_dict = {"input": input_tensor, "target": target_tensor, "size_average": size_average_bool, "reduce": reduce_bool, "reduction": reduction_str}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = np.array([[0.8, 0.6, 0.4, 0.2]], dtype=np.float32)
    target_tensor = np.array([[2, 3, -1, -1]], dtype=np.int64)
    size_average_bool = True
    reduce_bool = False
    reduction_str = 'mean'
    input_dict = {"input": input_tensor, "target": target_tensor, "size_average": size_average_bool, "reduce": reduce_bool, "reduction": reduction_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.multilabel_margin_loss"] = multilabel_margin_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.multilabel_margin_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.multilabel_margin_loss'.")

check_valid('torch.nn.functional.multilabel_margin_loss', generated_inputs['torch.nn.functional.multilabel_margin_loss'], lib="torch", suffix=0)
