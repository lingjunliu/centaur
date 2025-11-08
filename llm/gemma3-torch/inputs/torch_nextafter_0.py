
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def nextafter_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1.0, 2.0, 3.0])
    other1 = torch.tensor([2.0, 1.0, 4.0])
    out1 = torch.empty_like(input1)
    
    input_dict1 = {
        "input": input1,
        "other": other1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([-1.0, -2.0, -3.0])
    other2 = torch.tensor([-2.0, -1.0, -4.0])
    out2 = torch.empty_like(input2)
    
    input_dict2 = {
        "input": input2,
        "other": other2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    
    return list_of_inputs

generated_inputs["torch.nextafter"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nextafter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nextafter'.")


check_valid('torch.nextafter', generated_inputs['torch.nextafter'], lib="torch", suffix=0)
