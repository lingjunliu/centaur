
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def ge_inputs():
    list_of_inputs = []
    
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other = 2.0
    out = torch.empty(3, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    other = 2.5
    out = torch.empty((2, 2), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    other = 0.0
    out = torch.empty(3, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([5.0, -3.0, 0.0, -10.0]).numpy()
    other = -5.0
    out = torch.empty(4, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    other = 4.5
    out = torch.empty((2, 2, 2), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([100.0, 200.0, 300.0]).numpy()
    other = 150.0
    out = torch.empty(3, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([0.001, 0.002, 0.003]).numpy()
    other = 0.0015
    out = torch.empty(3, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([5.0]).numpy()
    other = 5.0
    out = torch.empty(1, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.ones((2, 2, 2, 2)).numpy()
    other = 0.5
    out = torch.empty((2, 2, 2, 2), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([-5.0, -10.0, -15.0, -20.0]).numpy()
    other = -12.0
    out = torch.empty(4, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.ge_2"] = ge_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.ge_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ge_2'.")


check_valid('torch.ge', generated_inputs['torch.ge_2'], lib="torch", suffix=2)
