
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def relu6_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-1.0, -2.0, -3.0, 0.0, 1.0, 2.0])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    inplace = True
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-5.0, -10.0, -15.0, -20.0])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([5.5, 5.9, 6.0, 6.1, 6.5, 7.0])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4, 4).astype(np.float32)
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([10.0, 20.0, 50.0, 100.0])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.zeros((3, 3))
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([3.5])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.relu6"] = relu6_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.relu6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.relu6'.")


check_valid('torch.nn.functional.relu6', generated_inputs['torch.nn.functional.relu6'], lib="torch", suffix=0)
