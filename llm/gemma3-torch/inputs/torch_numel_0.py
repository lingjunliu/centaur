
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def numel_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(2, 3, 4).numpy()
    list_of_inputs.append({"input": input1})
    
    input2 = torch.zeros(5, 5, 5, 5).numpy()
    list_of_inputs.append({"input": input2})
    
    input3 = torch.ones((1, 2, 3, 4, 5)).numpy()
    list_of_inputs.append({"input": input3})
    
    input4 = torch.randint(0, 10, (2, 2)).numpy()
    list_of_inputs.append({"input": input4})
    
    input5 = torch.randn(3).numpy()
    list_of_inputs.append({"input": input5})
    
    input6 = torch.zeros((10,)).numpy()
    list_of_inputs.append({"input": input6})
    
    input7 = torch.ones((2, 2, 2, 2, 2)).numpy()
    list_of_inputs.append({"input": input7})
    
    input8 = torch.randn(1, 1, 1, 1).numpy()
    list_of_inputs.append({"input": input8})
    
    input9 = torch.randint(0, 100, (4, 4, 4)).numpy()
    list_of_inputs.append({"input": input9})
    
    input10 = torch.zeros((100, 1)).numpy()
    list_of_inputs.append({"input": input10})

    return list_of_inputs

generated_inputs["torch.numel"] = numel_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.numel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.numel'.")


check_valid('torch.numel', generated_inputs['torch.numel'], lib="torch", suffix=0)
