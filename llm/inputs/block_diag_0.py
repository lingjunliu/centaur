
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def block_diag_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 and 3x3 matrices
    tensors = [np.array([[1, 2], [3, 4]]), np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single matrix
    tensors = [np.array([[1, 2, 3], [4, 5, 6]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D Tensors
    tensors = [np.array([1, 2, 3]), np.array([4, 5])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalars converted to 2D arrays
    tensors = [np.array([[1]]), np.array([[2]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.block_diag"] = block_diag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.block_diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.block_diag'.")

check_valid('torch.block_diag', generated_inputs['torch.block_diag'], lib="torch", suffix=0)
