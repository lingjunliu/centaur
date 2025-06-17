
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def column_stack_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two 1D tensors
    tensors1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    input_dict1 = {"tensors": tensors1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Two 2D tensors
    tensors2 = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    input_dict2 = {"tensors": tensors2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Three 1D tensors with different dtypes (int and float)
    tensors3 = [np.array([1, 2, 3]), np.array([4.0, 5.0, 6.0]), np.array([7, 8, 9])]
    input_dict3 = {"tensors": tensors3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Mixed dimensions (1D and 2D) -  This will fail
    # tensors4 = [np.array([1, 2, 3]), np.array([[4, 5, 6]])]
    # input_dict4 = {"tensors": tensors4}
    # list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Multiple 2D tensors
    tensors5 = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]]), np.array([[13, 14, 15], [16, 17, 18]])]
    input_dict5 = {"tensors": tensors5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: With negative values
    tensors6 = [np.array([-1, -2, -3]), np.array([-4, -5, -6])]
    input_dict6 = {"tensors": tensors6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Single 1D array
    tensors7 = [np.array([1, 2, 3])]
    input_dict7 = {"tensors": tensors7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Empty arrays
    tensors8 = [np.array([]), np.array([])]
    input_dict8 = {"tensors": tensors8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.column_stack"] = column_stack_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.column_stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.column_stack'.")

check_valid('torch.column_stack', generated_inputs['torch.column_stack'], lib="torch")
