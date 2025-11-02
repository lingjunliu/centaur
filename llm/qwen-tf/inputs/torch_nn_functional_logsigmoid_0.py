
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def logsigmoid_inputs():
    list_of_inputs = []
    
    input = torch.tensor([1.0, -1.0, 0.0, 10.0, -10.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[100.0, -100.0, 0.5],
                          [-0.5, 20.0, -20.0]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor(0.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.linspace(-5, 5, steps=2*3*4*5, dtype=torch.float32).reshape(2, 3, 4, 5).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.empty(0, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[-1.5, 0.0, 1.5],
                          [6.0, -6.0, 2.0]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    input = t.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    t = torch.linspace(-8, 8, steps=17, dtype=torch.float32)
    input = t[::2].numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.zeros(1, 2, 1, 3, 2, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.0, 0.0, 1e-8], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = (-10.0 * torch.rand(4, 5, dtype=torch.float64)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.logsigmoid"] = logsigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.logsigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.logsigmoid'.")


check_valid('torch.nn.functional.logsigmoid', generated_inputs['torch.nn.functional.logsigmoid'], lib="torch", suffix=0)
