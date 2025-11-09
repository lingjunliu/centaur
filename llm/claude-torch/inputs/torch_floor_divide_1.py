
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def floor_divide_inputs():
    list_of_inputs = []
    
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = torch.tensor([3.0, 4.0, 5.0]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[15.0, 25.0], [35.0, 45.0]]).numpy()
    other = torch.tensor([[3.0, 5.0], [7.0, 9.0]]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-10.0, -20.0, 30.0]).numpy()
    other = torch.tensor([3.0, 4.0, -5.0]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([10, 20, 30]).numpy()
    other = torch.tensor([3, 4, 5]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[[12.0, 15.0], [18.0, 21.0]], [[24.0, 27.0], [30.0, 33.0]]]).numpy()
    other = torch.tensor([[[3.0, 5.0], [6.0, 7.0]], [[8.0, 9.0], [10.0, 11.0]]]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([10.0, 20.0, 30.0, 40.0]).numpy()
    other = torch.tensor([2.0]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.ones((3, 4)).numpy() * 10
    other = torch.tensor([2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1000.0, 2000.0, 3000.0]).numpy()
    other = torch.tensor([7.0, 13.0, 17.0]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([0.5, 1.5, 2.5, 3.5]).numpy()
    other = torch.tensor([0.3, 0.4, 0.5, 0.6]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([100, 200, 300]).numpy()
    other = torch.tensor([11, 13, 17]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.floor_divide_1"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.floor_divide_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor_divide_1'.")


check_valid('torch.floor_divide', generated_inputs['torch.floor_divide_1'], lib="torch", suffix=1)
