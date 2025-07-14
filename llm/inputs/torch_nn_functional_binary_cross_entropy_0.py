
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def binary_cross_entropy_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    target_tensor = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight_tensor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'

    input_dict = {
        'input': input_tensor,
        'target': target_tensor,
        'weight': weight_tensor,
        'size_average': size_average,
        'reduce': reduce,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[0.2, 0.6], [0.8, 0.3]], dtype=np.float32)
    target_tensor = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    weight_tensor = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    size_average = False
    reduce = True
    reduction = 'sum'
    input_dict = {
        'input': input_tensor,
        'target': target_tensor,
        'weight': weight_tensor,
        'size_average': size_average,
        'reduce': reduce,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([0.4], dtype=np.float32)
    target_tensor = np.array([1.0], dtype=np.float32)
    weight_tensor = np.array([2.0], dtype=np.float32)
    size_average = False
    reduce = False
    reduction = 'none'
    input_dict = {
        'input': input_tensor,
        'target': target_tensor,
        'weight': weight_tensor,
        'size_average': size_average,
        'reduce': reduce,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([0.01, 0.99], dtype=np.float32)
    target_tensor = np.array([0.0, 1.0], dtype=np.float32)
    weight_tensor = np.array([1.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    input_dict = {
        'input': input_tensor,
        'target': target_tensor,
        'weight': weight_tensor,
        'size_average': size_average,
        'reduce': reduce,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([0.7, 0.2, 0.5], dtype=np.float32)
    target_tensor = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    weight_tensor = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    size_average = False
    reduce = True
    reduction = 'sum'

    input_dict = {
        'input': input_tensor,
        'target': target_tensor,
        'weight': weight_tensor,
        'size_average': size_average,
        'reduce': reduce,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[0.1, 0.9]], dtype=np.float32)
    target_tensor = np.array([[0.0, 1.0]], dtype=np.float32)
    weight_tensor = np.array([[1.0, 1.0]], dtype=np.float32)
    size_average = False
    reduce = False
    reduction = 'none'
    input_dict = {
        'input': input_tensor,
        'target': target_tensor,
        'weight': weight_tensor,
        'size_average': size_average,
        'reduce': reduce,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([0.3, 0.7], dtype=np.float32)
    target_tensor = np.array([0.0, 1.0], dtype=np.float32)
    weight_tensor = np.array([1.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'

    input_dict = {
        'input': input_tensor,
        'target': target_tensor,
        'weight': weight_tensor,
        'size_average': size_average,
        'reduce': reduce,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    input_tensor = np.array([0.2, 0.8, 0.4, 0.6], dtype=np.float32)
    target_tensor = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    weight_tensor = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    size_average = False
    reduce = True
    reduction = 'sum'

    input_dict = {
        'input': input_tensor,
        'target': target_tensor,
        'weight': weight_tensor,
        'size_average': size_average,
        'reduce': reduce,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[0.05, 0.95], [0.9, 0.1]], dtype=np.float32)
    target_tensor = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    weight_tensor = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    size_average = False
    reduce = False
    reduction = 'none'
    input_dict = {
        'input': input_tensor,
        'target': target_tensor,
        'weight': weight_tensor,
        'size_average': size_average,
        'reduce': reduce,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([0.6, 0.4], dtype=np.float32)
    target_tensor = np.array([1.0, 0.0], dtype=np.float32)
    weight_tensor = np.array([1.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'

    input_dict = {
        'input': input_tensor,
        'target': target_tensor,
        'weight': weight_tensor,
        'size_average': size_average,
        'reduce': reduce,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.array([0.999, 0.001], dtype=np.float32)
    target_tensor = np.array([1.0, 0.0], dtype=np.float32)
    weight_tensor = np.array([2.0, 0.5], dtype=np.float32)
    size_average = False
    reduce = False
    reduction = 'none'

    input_dict = {
        'input': input_tensor,
        'target': target_tensor,
        'weight': weight_tensor,
        'size_average': size_average,
        'reduce': reduce,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.binary_cross_entropy"] = binary_cross_entropy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.binary_cross_entropy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.binary_cross_entropy'.")

check_valid('torch.nn.functional.binary_cross_entropy', generated_inputs['torch.nn.functional.binary_cross_entropy'], lib="torch", suffix=0)
