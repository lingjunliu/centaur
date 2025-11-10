
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def minimum_inputs():
    list_of_inputs = []
    
    input_val = torch.tensor([1, 2, -1]).numpy()
    other = torch.tensor([3, 0, 4]).numpy()
    out = torch.tensor([]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([[1.5, 2.5], [3.5, 4.5]]).numpy()
    other = torch.tensor([[2.0, 1.0], [3.0, 5.0]]).numpy()
    out = torch.tensor([[]]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([-5.0, -3.0, -1.0]).numpy()
    other = torch.tensor([-2.0, -4.0, -6.0]).numpy()
    out = torch.tensor([]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    other = torch.tensor([[[2, 1], [4, 3]], [[6, 5], [8, 7]]]).numpy()
    out = torch.tensor([[[]]]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other = torch.tensor([2.5]).numpy()
    out = torch.tensor([]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.randn(10, 10).numpy()
    other = torch.randn(10, 10).numpy()
    out = torch.tensor([[]]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.zeros(5).numpy()
    other = torch.ones(5).numpy()
    out = torch.tensor([]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([1.1, 2.2, 3.3, 4.4]).numpy()
    other = torch.tensor([1.05, 2.25, 3.2, 4.5]).numpy()
    out = torch.tensor([]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    other = torch.tensor([[0, 1, 2]]).numpy()
    out = torch.tensor([[]]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([10, 20, 30]).numpy()
    other = torch.tensor([15, 18, 25]).numpy()
    out = torch.tensor([]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.minimum"] = minimum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.minimum'.")


check_valid('torch.minimum', generated_inputs['torch.minimum'], lib="torch", suffix=0)
