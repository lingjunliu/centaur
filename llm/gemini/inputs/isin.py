
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def isin_inputs():
    list_of_inputs = []

    elements = np.array([[1, 2], [3, 4]])
    test_elements = np.array([2, 3])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    elements = np.array([1, 2, 2, 3, 4, 5])
    test_elements = np.array([2, 4])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    elements = np.array([1, 2, 2, 3, 4, 5])
    test_elements = np.array([2, 4])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": True,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    elements = np.array([[1.5, 2.5], [3.5, 4.5]])
    test_elements = np.array([2.5, 3.5])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    elements = np.array([-1, -2, 0, 1, 2])
    test_elements = np.array([-2, 1])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    elements = np.array([1, 2])
    test_elements = 2
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    elements = 1
    test_elements = np.array([1,2,3])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    elements = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    test_elements = np.array([2, 3, 7])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = isin_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('isin', generated_inputs)
