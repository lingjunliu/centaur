
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def broadcast_tensors_inputs():
    list_of_inputs = []

    # Input 1: Simple broadcast
    tensors = [torch.tensor([1, 2, 3]).numpy(), torch.tensor([4]).numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: More complex broadcast
    tensors = [torch.randn(2, 1, 4).numpy(), torch.randn(3, 1, 1).numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero-dimensional tensors
    tensors = [torch.tensor(5).numpy(), torch.randn(2, 3, 1).numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different dtypes, all should be broadcastable
    tensors = [np.array(1), np.array([2, 3])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting to a higher dimension
    tensors = [np.random.randn(1, 3, 1), np.random.randn(3, 1)]
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
