
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

    # Input 3: np.bool_(True)
    input_dict = {"mode": np.bool_(True).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: np.bool_(False)
    input_dict = {"mode": np.bool_(False).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: bool(1)
    input_dict = {"mode": bool(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: bool(0)
    input_dict = {"mode": bool(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: not False
    input_dict = {"mode": not False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: not True
    input_dict = {"mode": not True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: np.array(True).item()
    input_dict = {"mode": bool(np.array(True).item())}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: np.array(False).item()
    input_dict = {"mode": bool(np.array(False).item())}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: True using np.array and astype
    input_dict = {"mode": np.array([1]).astype(bool)[0].item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: False using np.array and astype
    input_dict = {"mode": np.array([0]).astype(bool)[0].item()}
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
