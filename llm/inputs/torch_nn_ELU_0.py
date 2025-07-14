
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def elu_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    alpha_val = 1.0
    inplace_bool = False
    input_dict = {"alpha": alpha_val, "inplace": inplace_bool, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[-1.0, -0.5], [0.0, 0.5]])
    alpha_val = 0.5
    inplace_bool = True
    input_dict = {"alpha": alpha_val, "inplace": inplace_bool, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]])
    alpha_val = 2.0
    inplace_bool = False
    input_dict = {"alpha": alpha_val, "inplace": inplace_bool, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[-10.0, -5.0, 0.0], [5.0, 10.0, 15.0]])
    alpha_val = 0.1
    inplace_bool = True
    input_dict = {"alpha": alpha_val, "inplace": inplace_bool, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[-0.1, -0.05, 0.0], [0.05, 0.1, 0.15]])
    alpha_val = 10.0
    inplace_bool = False
    input_dict = {"alpha": alpha_val, "inplace": inplace_bool, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[-1.0, 0.0], [0.0, 1.0]])
    alpha_val = 1.5
    inplace_bool = True
    input_dict = {"alpha": alpha_val, "inplace": inplace_bool, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[-1.5, -0.75, 0.0], [0.75, 1.5, 2.25]])
    alpha_val = 0.75
    inplace_bool = False
    input_dict = {"alpha": alpha_val, "inplace": inplace_bool, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor
    input_tensor = np.random.randn(2, 3, 4)
    alpha_val = 1.0
    inplace_bool = True
    input_dict = {"alpha": alpha_val, "inplace": inplace_bool, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All negative values
    input_tensor = np.array([-5.0, -2.5, -1.0, -0.5])
    alpha_val = 0.25
    inplace_bool = False
    input_dict = {"alpha": alpha_val, "inplace": inplace_bool, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All positive values
    input_tensor = np.array([5.0, 2.5, 1.0, 0.5])
    alpha_val = 0.8
    inplace_bool = True
    input_dict = {"alpha": alpha_val, "inplace": inplace_bool, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 : 1D tensor with mixed values and different alpha
    input_tensor = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    alpha_val = 3.0
    inplace_bool = False
    input_dict = {"alpha": alpha_val, "inplace": inplace_bool, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ELU"] = elu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ELU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ELU'.")

check_valid('torch.nn.ELU', generated_inputs['torch.nn.ELU'], lib="torch", suffix=0)
