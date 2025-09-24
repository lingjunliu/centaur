
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clip_inputs():
    list_of_inputs = []
    
    # Input 1
    input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0]).numpy()
    min_val = 0.0
    max_val = 1.0
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_tensor = torch.tensor([[-1.0, 0.0], [1.0, 2.0]]).numpy()
    min_val = -0.5
    max_val = 1.5
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_tensor = torch.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]).numpy()
    min_val = -1.0
    max_val = 2.0
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(2, 3, 4).numpy()
    min_val = -0.7
    max_val = 0.9
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randint(-5, 5, (5,)).float().numpy()
    min_val = -2.5
    max_val = 2.5
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_tensor = torch.arange(-10, 10, 2).float().numpy()
    min_val = -5.0
    max_val = 5.0
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_tensor = torch.randn(1, 1, 1).numpy()
    min_val = -0.1
    max_val = 0.1
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.clip"] = clip_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.clip' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clip'.")

check_valid('torch.clip', generated_inputs['torch.clip'], lib="torch", suffix=0)
