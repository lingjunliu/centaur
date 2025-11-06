
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logsigmoid_inputs():
    list_of_inputs = []
    
    input = torch.randn(2, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[-10.0, -1.0, 0.0, 1.0, 10.0]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.arange(-5.0, 5.0, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor(3.14, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.randn(2, 0, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.linspace(-100, 100, steps=9, dtype=torch.float32).reshape(3, 3).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.randn(1, 3, 32, 32, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([1e-6, 1e-12, -1e-6, -1e2, 1e2], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.randn(4, 5, dtype=torch.float32).t().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.full((2, 2, 2), fill_value=-700.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    return list_of_inputs

generated_inputs["torch.nn.LogSigmoid"] = logsigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LogSigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LogSigmoid'.")


check_valid('torch.nn.LogSigmoid', generated_inputs['torch.nn.LogSigmoid'], lib="torch", suffix=0)
