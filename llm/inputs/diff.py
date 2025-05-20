
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def diff_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input1 = np.array([1, 3, 2])
    input_dict1 = {"input": input1, "n": 1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, different dim, higher order difference
    input2 = np.array([[1, 2, 3], [3, 4, 5]])
    input_dict2 = {"input": input2, "n": 2, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor with prepend and append
    input3 = np.array([1, 3, 2])
    prepend3 = np.array([0])
    append3 = np.array([4])
    input_dict3 = {"input": input3, "n": 1, "dim": 0, "prepend": prepend3, "append": append3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor, negative values
    input4 = np.array([[[1, 2, 3], [-4, -5, -6]], [[7, 8, 9], [-10, -11, -12]]])
    input_dict4 = {"input": input4, "n": 1, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Empty tensor
    input5 = np.array([])
    input_dict5 = {"input": input5, "n": 1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: float tensor
    input6 = np.array([1.5, 2.5, 3.5])
    input_dict6 = {"input": input6, "n": 1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: n = 0, should return original array
    input7 = np.array([1, 2, 3])
    input_dict7 = {"input": input7, "n": 0, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = diff_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('diff', generated_inputs)
