
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def isposinf_inputs():
    list_of_inputs = []

    # Input 1: Basic test with positive infinity
    input_tensor = torch.tensor([float('inf'), 1.0, -float('inf'), 0.0]).numpy()
    out_tensor = torch.tensor([False, False, False, False]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tensor with no positive infinity
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    out_tensor = torch.tensor([False, False, False, False]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensor with only positive infinity
    input_tensor = torch.tensor([float('inf'), float('inf'), float('inf')]).numpy()
    out_tensor = torch.tensor([False, False, False]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Multi-dimensional tensor
    input_tensor = torch.tensor([[1.0, float('inf')], [float('-inf'), 2.0]]).numpy()
    out_tensor = torch.tensor([[False, False], [False, False]]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with a mix of numbers and positive infinity
    input_tensor = torch.tensor([float('inf'), 0.0, 1.5, float('inf')]).numpy()
    out_tensor = torch.tensor([False, False, False, False]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.isposinf"] = isposinf_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.isposinf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isposinf'.")

check_valid('torch.isposinf', generated_inputs['torch.isposinf'], lib="torch", suffix=0)
