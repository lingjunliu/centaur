
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def logical_xor_inputs():
    list_of_inputs = []

    input1 = np.array([True, False, True, False])
    input2 = np.array([True, True, False, False])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([[True, False], [True, False]])
    input2 = np.array([[True, True], [False, False]])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([True, False, True])
    input2 = np.array(False)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array(True)
    input2 = np.array([True, False, True])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([[True, False, True], [False, True, False]])
    input2 = np.array([[False, True, False], [True, False, True]])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.array([[[True, False], [True, False]], [[False, True], [False, True]]])
    input2 = np.array([[[True, True], [False, False]], [[True, False], [True, False]]])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.array([True])
    input2 = np.array(False)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = logical_xor_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('logical_xor', generated_inputs)
