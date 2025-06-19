
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

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

    elements = np.array([1, 2, 3, 4, 5, 1, 2, 3])
    test_elements = np.array([2, 3, 4])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": True,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    elements = np.array([[1.1, 2.2], [3.3, 4.4]])
    test_elements = np.array([2.2, 3.3])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    elements = np.array([-1, -2, 3, 4])
    test_elements = np.array([-2, 4])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.isin"] = isin_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.isin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isin'.")

check_valid('torch.isin', generated_inputs['torch.isin'], lib="torch")
