
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def cartesian_prod_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two 1D tensors
    tensors = [torch.tensor([1, 2]).numpy(), torch.tensor([3, 4]).numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Three tensors, mixed dimensions. Removing the 2D tensor
    tensors = [torch.tensor([1, 2]).numpy(), torch.tensor([3, 4]).numpy(), torch.tensor([7]).numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensors with different dtypes (integers and floats)
    tensors = [torch.tensor([1, 2]).numpy(), torch.tensor([3.0, 4.0]).numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Larger tensors
    tensors = [torch.tensor([1, 2, 3]).numpy(), torch.tensor([4, 5, 6]).numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.cartesian_prod"] = cartesian_prod_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cartesian_prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cartesian_prod'.")

check_valid('torch.cartesian_prod', generated_inputs['torch.cartesian_prod'], lib="torch", suffix=0)
