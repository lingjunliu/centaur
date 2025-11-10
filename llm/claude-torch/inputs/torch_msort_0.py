
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def msort_inputs():
    list_of_inputs = []
    
    input = torch.tensor([[-0.1321, 0.4370, -1.2631, -1.1289],
                          [-2.0527, -1.1250, 0.2275, 0.3077],
                          [-0.0881, -0.1259, -0.5495, 1.0284]]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([5.0, 2.0, 8.0, 1.0, 9.0]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(4, 3, 2).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[-5.0, -2.0], [-9.0, -1.0], [-3.0, -7.0]]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[3.0, 7.0, 1.0], [9.0, 2.0, 5.0]]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[0.0, -1.0, 2.0], [1.0, 0.0, -2.0], [-1.0, 2.0, 0.0]]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([42.0]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[10, 5, 8], [3, 15, 1], [7, 2, 12]]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4, 5).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.msort"] = msort_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.msort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.msort'.")


check_valid('torch.msort', generated_inputs['torch.msort'], lib="torch", suffix=0)
