
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def neg_inputs():
    list_of_inputs = []
    
    # Input 1
    input_tensor = torch.tensor([1.0, -2.0, 3.0]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(2, 3).numpy()
    out_tensor = torch.zeros(2, 3).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.zeros(5).numpy()
    out_tensor = torch.ones(5).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.ones(2, 2, 2).numpy()
    out_tensor = torch.zeros(2, 2, 2).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.arange(-5, 5, dtype=torch.float32).numpy()
    out_tensor = torch.zeros(10).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_tensor = torch.tensor([float('inf'), float('-inf'), float('nan')]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_tensor = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    out_tensor = torch.tensor([0, 0, 0], dtype=torch.int32).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_tensor = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    out_tensor = torch.tensor([0, 0, 0], dtype=torch.int64).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.neg"] = neg_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.neg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.neg'.")

check_valid('torch.neg', generated_inputs['torch.neg'], lib="torch", suffix=0)
