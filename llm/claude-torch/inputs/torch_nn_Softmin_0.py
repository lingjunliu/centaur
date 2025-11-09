
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def softmin_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dim = 1
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    dim = 0
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dim = 0
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    dim = 2
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(4, 5, 6).astype(np.float32)
    dim = 1
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4, 5).astype(np.float32)
    dim = 3
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
    dim = 1
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    dim = 0
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(3, 2, 4).astype(np.float32)
    dim = 0
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    dim = 1
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Softmin"] = softmin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Softmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softmin'.")


check_valid('torch.nn.Softmin', generated_inputs['torch.nn.Softmin'], lib="torch", suffix=0)
