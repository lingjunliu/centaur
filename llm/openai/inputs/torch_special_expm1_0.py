
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def expm1_inputs():
    list_of_inputs = []
    
    input = torch.tensor([-1.0, 0.0, 1.0, 10.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[-1e-2, 2e-2, -3e-1],
                          [4.5, -5.0, 6.25]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.linspace(-1, 1, steps=12, dtype=torch.float16).reshape(2, 2, 3).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor(0.0, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([-100.0, -50.0, -0.1], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([1+1j, -1+2j, -3j, 0+0j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[-10+0j, 0+0j],
                          [3.5-4.2j, -2.0+2.0j]], dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.arange(-6, 6, dtype=torch.float32).reshape(1, 2, 3, 2).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.empty((0,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([float('inf'), float('-inf'), float('nan'), 1.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([1e-8, -1e-8, 1e-12, -1e-12], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([20.0, 50.0, 88.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    return list_of_inputs

generated_inputs["torch.special.expm1"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.expm1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.expm1'.")


check_valid('torch.special.expm1', generated_inputs['torch.special.expm1'], lib="torch", suffix=0)
