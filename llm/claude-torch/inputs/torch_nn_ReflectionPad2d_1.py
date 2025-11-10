
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reflectionpad2d_inputs():
    list_of_inputs = []
    
    padding = 1
    input_tensor = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 2
    input_tensor = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 1
    input_tensor = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 1
    input_tensor = np.random.randn(1, 3, 4, 4).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 2
    input_tensor = np.random.randn(2, 2, 5, 5).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 1
    input_tensor = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 1
    input_tensor = np.random.randn(1, 1, 4, 6).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 3
    input_tensor = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 1
    input_tensor = np.random.randn(1, 1, 10, 10).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 2
    input_tensor = np.random.randn(4, 3, 6, 6).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad2d_1"] = reflectionpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReflectionPad2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad2d_1'.")


check_valid('torch.nn.ReflectionPad2d', generated_inputs['torch.nn.ReflectionPad2d_1'], lib="torch", suffix=1)
