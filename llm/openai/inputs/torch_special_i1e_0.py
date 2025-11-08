
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def special_i1e_inputs():
    list_of_inputs = []
    
    x = torch.tensor([0.0, 1.0, -1.0, 2.5, -3.5], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x}))
    
    x = torch.tensor([[0.1, -0.2, 3.0], [-4.0, 5.5, -6.5]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x}))
    
    x = torch.tensor(0.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x}))
    
    x = torch.arange(-6, 6, dtype=torch.float16).reshape(3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x}))
    
    x = torch.linspace(0, 50, steps=7, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x}))
    
    x = torch.tensor([float('nan'), float('inf'), float('-inf'), 0.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x}))
    
    x = torch.tensor([], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x}))
    
    x = torch.arange(24, dtype=torch.float64).view(4, 6)[::2, ::3].numpy()
    list_of_inputs.append(copy.deepcopy({"x": x}))
    
    x = torch.tensor([-0.1, -1.0, -10.0, -50.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x}))
    
    x = torch.ones((2, 1, 3, 1), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x}))
    
    x = torch.tensor([1e-30, -1e-30, 1e-10, -1e-10], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x}))
    
    x = torch.arange(-8, 8, dtype=torch.float16).reshape(4, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x}))
    
    return list_of_inputs

generated_inputs["torch.special.i1e"] = special_i1e_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.i1e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.i1e'.")


check_valid('torch.special.i1e', generated_inputs['torch.special.i1e'], lib="torch", suffix=0)
