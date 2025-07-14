
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def broadcast_tensors_inputs():
    list_of_inputs = []

    # Input 1: Simple broadcast
    tensors = [np.array([1, 2, 3]), np.array([[4], [5], [6]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcast to a higher dimension
    tensors = [np.array([1]), np.array([[2, 3]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: No broadcast needed
    tensors = [np.array([1, 2]), np.array([3, 4])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: More complex broadcast
    tensors = [np.array([1, 2, 3]), np.array([4])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional arrays
    tensors = [np.array([[1, 2], [3, 4]]), np.array([5, 6])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different dtypes, implicit conversion
    tensors = [np.array([1, 2, 3], dtype=np.int32), np.array([4.0])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Larger tensors
    tensors = [np.random.rand(2, 3, 4), np.random.rand(4)]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.broadcast_tensors"] = broadcast_tensors_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.broadcast_tensors' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.broadcast_tensors'.")

check_valid('torch.broadcast_tensors', generated_inputs['torch.broadcast_tensors'], lib="torch", suffix=0)
