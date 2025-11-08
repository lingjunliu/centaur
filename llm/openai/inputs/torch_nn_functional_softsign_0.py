
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def softsign_inputs():
    list_of_inputs = []
    
    input = torch.tensor(0.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = (torch.randn((2, 3), dtype=torch.float64) * 100.0).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.randn((2, 2, 3), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.linspace(-5, 5, steps=48, dtype=torch.float32).reshape(1, 3, 4, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    base = torch.arange(30, dtype=torch.float32).reshape(5, 6).numpy()
    input = base[::-2, 1::2]
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([1e-12, -1e-12, 1e12, -1e12], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([float('inf'), -float('inf'), 0.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([float('nan'), 2.0, -2.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.randn((2, 1, 2, 1, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.full((3, 3), -2.5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    base = torch.randn((3, 5), dtype=torch.float64).numpy()
    input = base.T
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.softsign"] = softsign_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.softsign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.softsign'.")


check_valid('torch.nn.functional.softsign', generated_inputs['torch.nn.functional.softsign'], lib="torch", suffix=0)
