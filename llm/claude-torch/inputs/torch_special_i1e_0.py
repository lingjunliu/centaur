
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def i1e_inputs():
    list_of_inputs = []
    
    input_val = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([-5.0, 0.0, 5.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    out = torch.empty(2, 2).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]]).numpy()
    out = torch.empty(2, 2, 2).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor(2.5).numpy()
    out = torch.empty(()).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([10.0, 20.0, 30.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([0.001, 0.01, 0.1]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]).numpy()
    out = torch.empty(2, 3).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([0.0]).numpy()
    out = torch.empty(1).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.ones((2, 2, 2, 2)).numpy()
    out = torch.empty(2, 2, 2, 2).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.i1e"] = i1e_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.i1e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.i1e'.")


check_valid('torch.special.i1e', generated_inputs['torch.special.i1e'], lib="torch", suffix=0)
