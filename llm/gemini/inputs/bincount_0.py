
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def bincount_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with no weights and default minlength
    input1 = np.array([2, 1, 0, 2, 0], dtype=np.int64)
    input_dict1 = {"input": input1, "weights": None, "minlength": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: With weights
    input2 = np.array([2, 1, 0, 2, 0], dtype=np.int64)
    weights2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    input_dict2 = {"input": input2, "weights": weights2, "minlength": 0}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: With minlength
    input3 = np.array([2, 1, 0, 2, 0], dtype=np.int64)
    input_dict3 = {"input": input3, "weights": None, "minlength": 5}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Empty input
    input4 = np.array([], dtype=np.int64)
    input_dict4 = {"input": input4, "weights": None, "minlength": 0}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Test case 5: Empty input with minlength
    input5 = np.array([], dtype=np.int64)
    input_dict5 = {"input": input5, "weights": None, "minlength": 3}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6: Different input dtype
    input6 = np.array([2, 1, 0, 2, 0], dtype=np.int32)
    input_dict6 = {"input": input6, "weights": None, "minlength": 0}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Test case 7: Weights with different dtype
    input7 = np.array([2, 1, 0, 2, 0], dtype=np.int64)
    weights7 = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    input_dict7 = {"input": input7, "weights": weights7, "minlength": 0}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Test case 8: Larger values in input
    input8 = np.array([10, 5, 2, 10, 0], dtype=np.int64)
    input_dict8 = {"input": input8, "weights": None, "minlength": 0}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.bincount"] = bincount_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bincount'.")

check_valid('torch.bincount', generated_inputs['torch.bincount'], lib="torch")
