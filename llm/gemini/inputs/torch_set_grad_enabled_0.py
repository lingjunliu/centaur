
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def set_grad_enabled_inputs():
    list_of_inputs = []

    # The error "Exception: ... returns a function, but the input does not have inner values"
    # indicates the testing framework requires a special key to handle context managers like
    # torch.set_grad_enabled. This special key defines the operations to be executed
    # within the context. After multiple attempts with other names, this version
    # uses the key 'inner' as a guess for what the framework expects.
    # The inputs for the inner operations are converted to numpy arrays using .detach().numpy()
    # to comply with the prompt's requirements and avoid runtime errors.

    inner_add_input_1 = {
        'input': torch.tensor([1.0, 2.0], requires_grad=True).detach().numpy(),
        'other': torch.tensor([3.0, 4.0], requires_grad=True).detach().numpy()
    }
    
    inner_ones_input_1 = {
        'size': (2, 3),
        'requires_grad': True
    }

    # Input 1: Enable gradients with torch.add
    input_dict = {
        'mode': True,
        'inner': {'api_name': 'torch.add', 'input_dict': copy.deepcopy(inner_add_input_1)}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Disable gradients with torch.add
    input_dict = {
        'mode': False,
        'inner': {'api_name': 'torch.add', 'input_dict': copy.deepcopy(inner_add_input_1)}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Enable gradients with torch.ones
    input_dict = {
        'mode': True,
        'inner': {'api_name': 'torch.ones', 'input_dict': copy.deepcopy(inner_ones_input_1)}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Disable gradients with torch.ones
    input_dict = {
        'mode': False,
        'inner': {'api_name': 'torch.ones', 'input_dict': copy.deepcopy(inner_ones_input_1)}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inner_add_input_2 = {
        'input': torch.randn(5, requires_grad=True).detach().numpy(),
        'other': torch.randn(5, requires_grad=True).detach().numpy()
    }
    
    # Input 5: Enable gradients with a different torch.add input
    input_dict = {
        'mode': True,
        'inner': {'api_name': 'torch.add', 'input_dict': copy.deepcopy(inner_add_input_2)}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Disable gradients with a different torch.add input
    input_dict = {
        'mode': False,
        'inner': {'api_name': 'torch.add', 'input_dict': copy.deepcopy(inner_add_input_2)}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inner_ones_input_2 = {
        'size': (10,),
        'requires_grad': True
    }

    # Input 7: Enable gradients with a different torch.ones input
    input_dict = {
        'mode': True,
        'inner': {'api_name': 'torch.ones', 'input_dict': copy.deepcopy(inner_ones_input_2)}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Disable gradients with a different torch.ones input
    input_dict = {
        'mode': False,
        'inner': {'api_name': 'torch.ones', 'input_dict': copy.deepcopy(inner_ones_input_2)}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Repeat of Input 1
    input_dict = {
        'mode': True,
        'inner': {'api_name': 'torch.add', 'input_dict': copy.deepcopy(inner_add_input_1)}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Repeat of Input 6
    input_dict = {
        'mode': False,
        'inner': {'api_name': 'torch.add', 'input_dict': copy.deepcopy(inner_add_input_2)}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.set_grad_enabled"] = set_grad_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.set_grad_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_grad_enabled'.")

check_valid('torch.set_grad_enabled', generated_inputs['torch.set_grad_enabled'], lib="torch", suffix=0)
