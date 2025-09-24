
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def row_stack_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensors
    tensors = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int tensors
    tensors = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[5, 6], [7, 8]], dtype=np.int32)]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shapes along dim 1
    tensors = [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensors
    tensors = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensors
    tensors = [np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
               np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values
    tensors = [np.array([[-1.0, 2.0], [3.0, -4.0]]), np.array([[5.0, -6.0], [-7.0, 8.0]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mix of int and float, upcasted
    tensors = [np.array([[1, 2], [3, 4]]), np.array([[5.0, 6.0], [7.0, 8.0]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = row_stack_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('row_stack', generated_inputs)
