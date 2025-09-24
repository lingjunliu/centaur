
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def threshold_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([-1.0, 0.0, 1.0, 2.0])
    threshold_val = 0.5
    value_val = 0.0
    inplace_bool = False
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val, "inplace": inplace_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([-1.0, 0.0, 1.0, 2.0])
    threshold_val = 1.0
    value_val = -1.0
    inplace_bool = True
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val, "inplace": inplace_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    threshold_val = 2.5
    value_val = 0.0
    inplace_bool = False
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val, "inplace": inplace_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    threshold_val = 3.0
    value_val = 1.0
    inplace_bool = True
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val, "inplace": inplace_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    threshold_val = 4.5
    value_val = -2.0
    inplace_bool = False
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val, "inplace": inplace_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    threshold_val = 7.0
    value_val = 3.0
    inplace_bool = True
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val, "inplace": inplace_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
    threshold_val = -3.0
    value_val = 10.0
    inplace_bool = False
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val, "inplace": inplace_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
    threshold_val = -2.0
    value_val = 5.0
    inplace_bool = True
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val, "inplace": inplace_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    threshold_val = 0.3
    value_val = -0.1
    inplace_bool = False
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val, "inplace": inplace_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    threshold_val = 0.4
    value_val = -0.2
    inplace_bool = True
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val, "inplace": inplace_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_tensor = np.array([100.0, 200.0, 300.0])
    threshold_val = 150.0
    value_val = 50.0
    inplace_bool = False
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val, "inplace": inplace_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.threshold"] = threshold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.threshold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.threshold'.")

check_valid('torch.nn.functional.threshold', generated_inputs['torch.nn.functional.threshold'], lib="torch", suffix=0)
