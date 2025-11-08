
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def dropout_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([1.0, -2.0, 3.0, -4.5], dtype=np.float32)
    p = 0.5
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = np.ones((2, 3), dtype=np.float64)
    p = 0.2
    training = True
    inplace = True
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = np.random.randn(2, 3, 4).astype(np.float32)
    p = 0.0
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = np.ones((2, 3, 4, 4), dtype=np.float32) * 5.0
    p = 0.9
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = np.array(5.0, dtype=np.float32)
    p = 0.7
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = np.random.uniform(-1, 1, size=(1, 2, 1, 3, 4)).astype(np.float32)
    p = 0.3
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = np.empty((0, 3), dtype=np.float32)
    p = 0.5
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = np.array([np.nan, np.inf, -np.inf, 0.0, -1.0], dtype=np.float64)
    p = 0.4
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = np.random.randn(4, 5).astype(np.float32)
    p = 0.6
    training = False
    inplace = True
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = (np.random.rand(10, 10).astype(np.float32) - 0.5) * 20.0
    p = 0.05
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = np.array([1000.0, -1000.0, 0.123, -0.987], dtype=np.float64)
    p = 0.9
    training = True
    inplace = True
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = np.array([[[-1.0, 2.0], [3.5, -4.5]],
                          [[0.0, -0.1], [0.2, -0.3]]], dtype=np.float32)
    p = 0.33
    training = False
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.dropout"] = dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.dropout'.")


check_valid('torch.nn.functional.dropout', generated_inputs['torch.nn.functional.dropout'], lib="torch", suffix=0)
