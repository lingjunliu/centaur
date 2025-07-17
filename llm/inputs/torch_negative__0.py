
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def negative_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, positive values
    input_tensor = np.array([1, 2, 3], dtype=np.int32)

    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor, negative values
    input_tensor = np.array([-1, -2, -3], dtype=np.int64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor, mixed values
    input_tensor = np.array([-1, 0, 3], dtype=np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensor, positive values
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensor, negative values
    input_tensor = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D tensor, mixed values
    input_tensor = np.array([[-1, 2], [3, -4]], dtype=np.float64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor, positive values
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor, negative values
    input_tensor = np.array([[[ -1, -2], [-3, -4]], [[-5, -6], [-7, -8]]], dtype=np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: different dtypes
    input_tensor = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: out tensor with pre-allocated shape
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.negative_"] = negative_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.negative_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.negative_'.")

check_valid('torch.negative_', generated_inputs['torch.negative_'], lib="torch", suffix=0)
