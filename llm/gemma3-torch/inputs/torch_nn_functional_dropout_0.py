
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def dropout_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    input_dict1 = {
        "input": input1,
        "p": 0.5,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict2 = {
        "input": input2,
        "p": 0.2,
        "training": False,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "p": 0.8,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([-1.0, -2.0, -3.0])
    input_dict4 = {
        "input": input4,
        "p": 0.1,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.zeros((5, 5))
    input_dict5 = {
        "input": input5,
        "p": 0.9,
        "training": False,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.ones((1, 2, 3, 4))
    input_dict6 = {
        "input": input6,
        "p": 0.7,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([0.0, 0.0, 0.0])
    input_dict7 = {
        "input": input7,
        "p": 0.3,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1.1, 2.2, 3.3])
    input_dict8 = {
        "input": input8,
        "p": 0.6,
        "training": False,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(10)
    input_dict9 = {
        "input": input9,
        "p": 0.4,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[1.0], [2.0], [3.0]])
    input_dict10 = {
        "input": input10,
        "p": 0.0,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.dropout"] = dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.dropout'.")


check_valid('torch.nn.functional.dropout', generated_inputs['torch.nn.functional.dropout'], lib="torch", suffix=0)
