
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def le_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([1, 2, 3])
    other = 2.0
    out = np.array([True, False, False])
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.array([[1, 2], [3, 4]])
    other = 3.0
    out = np.array([[True, True], [True, False]])
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.array([-1, 0, 1])
    other = 0.0
    out = np.array([True, True, False])
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = np.array([5, 5, 5])
    other = 5.0
    out = np.array([True, True, True])
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.array([1.5, 2.5, 3.5])
    other = 2.5
    out = np.array([True, True, False])
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = np.array([[-1.0, 2.0], [3.0, -4.0]])
    other = 0.0
    out = np.array([[True, False], [False, True]])
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = np.array([1, 2, 3], dtype=np.int64)
    other = 2.0
    out = np.array([True, False, False])
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other = 2.0
    out = np.array([True, False, False])
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = np.array([1, 2, 3])
    other = -1.0
    out = np.array([False, False, False])
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    other = 5.0
    out = np.array([[[ True,  True], [ True,  True]],[[ True, False],[False, False]]])
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.le_2"] = le_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.le_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.le_2'.")

check_valid('torch.le', generated_inputs['torch.le_2'], lib="torch", suffix=2)
