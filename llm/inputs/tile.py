
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def torch_tile_inputs():
    list_of_inputs = []

    # Example 1: 1D tensor, simple repetition
    input1 = np.array([1, 2, 3])
    dims1 = (2,)
    input_dict1 = {"input": input1, "dims": dims1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: 2D tensor, repetition in both dimensions
    input2 = np.array([[1, 2], [3, 4]])
    dims2 = (2, 2)
    input_dict2 = {"input": input2, "dims": dims2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: 3D tensor, repetition in only some dimensions
    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dims3 = (1, 2, 1)
    input_dict3 = {"input": input3, "dims": dims3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: Integer tensor
    input4 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dims4 = (2, 1)
    input_dict4 = {"input": input4, "dims": dims4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: Float tensor
    input5 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    dims5 = (1, 2)
    input_dict5 = {"input": input5, "dims": dims5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Example 6: Tensor with negative values
    input6 = np.array([[-1, 2], [3, -4]])
    dims6 = (2, 2)
    input_dict6 = {"input": input6, "dims": dims6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Example 7: Different dimensions
    input7 = np.array([1, 2, 3, 4])
    dims7 = (4,)
    input_dict7 = {"input": input7, "dims": dims7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Example 8: Different dimensions
    input8 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dims8 = (2, 1, 3)
    input_dict8 = {"input": input8, "dims": dims8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Example 9: Different dtype
    input9 = np.array([1, 2, 3], dtype=np.int64)
    dims9 = (3,)
    input_dict9 = {"input": input9, "dims": dims9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs = torch_tile_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('tile', generated_inputs)
