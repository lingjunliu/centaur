
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def set_grad_enabled_inputs():
    list_of_inputs = []

    # Input 1: True
    input_dict = {"mode": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: False
    input_dict = {"mode": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: numpy boolean True
    input_dict = {"mode": np.array(True, dtype=bool).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: numpy boolean False
    input_dict = {"mode": np.array(False, dtype=bool).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: numpy array of boolean True
    input_dict = {"mode": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numpy array of boolean False
    input_dict = {"mode": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: numpy bool with type
    input_dict = {"mode": np.array(True, dtype=bool).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: numpy bool with type
    input_dict = {"mode": np.array(False, dtype=bool).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: different numpy boolean format
    input_dict = {"mode": np.array([True], dtype=bool)[0].item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: another different numpy boolean format
    input_dict = {"mode": np.array([False], dtype=bool)[0].item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.set_grad_enabled"] = set_grad_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.set_grad_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_grad_enabled'.")

check_valid('torch.set_grad_enabled', generated_inputs['torch.set_grad_enabled'], lib="torch", suffix=0)
