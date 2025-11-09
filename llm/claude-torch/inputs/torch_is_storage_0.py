
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def is_storage_inputs():
    list_of_inputs = []
    
    # Input 1: 1D tensor
    obj = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor
    obj = torch.ones((2, 3)).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor
    obj = torch.randn(2, 3, 4).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: scalar tensor
    obj = torch.tensor(5.0).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: negative values
    obj = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: integer tensor
    obj = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: empty tensor
    obj = torch.tensor([]).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor
    obj = torch.zeros((2, 3, 4, 5)).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: boolean tensor
    obj = torch.tensor([True, False, True]).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: large 2D tensor
    obj = torch.randn(100, 100).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: mixed positive and negative values
    obj = torch.tensor([[1.0, -2.0], [3.0, -4.0]]).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: zeros tensor
    obj = torch.zeros(10).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_storage"] = is_storage_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.is_storage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_storage'.")


check_valid('torch.is_storage', generated_inputs['torch.is_storage'], lib="torch", suffix=0)
