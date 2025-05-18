
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def dot_inputs():
    list_of_inputs = []

    input1 = np.array([1, 2, 3])
    input2 = np.array([4, 5, 6])
    input_dict = {"input": input1, "tensor": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([1.5, 2.5, 3.5])
    input2 = np.array([4.5, 5.5, 6.5])
    input_dict = {"input": input1, "tensor": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([-1, -2, -3])
    input2 = np.array([4, 5, 6])
    input_dict = {"input": input1, "tensor": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([0, 0, 0])
    input2 = np.array([1, 2, 3])
    input_dict = {"input": input1, "tensor": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([1, 1, 1])
    input2 = np.array([1, 1, 1])
    input_dict = {"input": input1, "tensor": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = dot_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('dot', list_of_inputs)
