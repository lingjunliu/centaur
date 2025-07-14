
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def dirac_inputs():
    list_of_inputs = []

    # Input 1: 3D tensor, no offset
    tensor = np.ones((3, 3, 3), dtype=np.float32) + 0.1
    offset = 0
    input_dict = {"tensor": tensor, "offset": offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D tensor, no offset
    tensor = np.ones((3, 3, 3, 3), dtype=np.float32) + 0.1
    offset = 0
    input_dict = {"tensor": tensor, "offset": offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 5D tensor, no offset
    tensor = np.ones((3, 3, 3, 3, 3), dtype=np.float32) + 0.1
    offset = 0
    input_dict = {"tensor": tensor, "offset": offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor, offset 1
    tensor = np.ones((4, 3, 3), dtype=np.float32) + 0.1
    offset = 1
    input_dict = {"tensor": tensor, "offset": offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D tensor, offset 1
    tensor = np.ones((4, 3, 3, 3), dtype=np.float32) + 0.1
    offset = 1
    input_dict = {"tensor": tensor, "offset": offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 5D tensor, offset 1
    tensor = np.ones((3, 2, 2, 2, 2), dtype=np.float32) + 0.1
    offset = 1
    input_dict = {"tensor": tensor, "offset": offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor, different shape, offset 0
    tensor = np.ones((2, 4, 5), dtype=np.float32) + 0.1
    offset = 0
    input_dict = {"tensor": tensor, "offset": offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor, different shape, offset 1
    tensor = np.ones((2, 3, 4, 5), dtype=np.float32) + 0.1
    offset = 1
    input_dict = {"tensor": tensor, "offset": offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D tensor, different shape, offset 0
    tensor = np.ones((2, 2, 3, 4, 5), dtype=np.float32) + 0.1
    offset = 0
    input_dict = {"tensor": tensor, "offset": offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D tensor, smaller size, offset 0
    tensor = np.ones((1, 1, 1), dtype=np.float32) + 0.1
    offset = 0
    input_dict = {"tensor": tensor, "offset": offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.init.dirac_"] = dirac_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.init.dirac_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.dirac_'.")

check_valid('torch.nn.init.dirac_', generated_inputs['torch.nn.init.dirac_'], lib="torch", suffix=0)
