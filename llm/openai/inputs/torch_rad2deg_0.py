
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def rad2deg_inputs():
    list_of_inputs = []
    
    input = torch.tensor([0.0, 1.5707963, 3.1415927, -3.1415927], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([[0.0, -1.5707963267948966],
                          [3.141592653589793, 6.283185307179586]], dtype=torch.float64).numpy()
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor(1.5707963267948966, dtype=torch.float64).numpy()
    out = np.empty((), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = (torch.arange(-6, 6, dtype=torch.float32).reshape(2, 2, 3) / 10.0).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.empty((0,), dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.empty((2, 0, 3), dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([float('nan'), float('inf'), float('-inf'), 0.7853982], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([[[[0.0, 0.5235988, -0.5235988]],
                           [[1.0471976, -1.0471976, 3.1415927]]]], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([[0.1, -0.2, 0.3],
                          [-0.4, 0.5, -0.6]], dtype=torch.float16).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([[[100.0, -100.0], [1e-6, -1e-6]],
                          [[123456.789, -123456.789], [6.283185307179586, -6.283185307179586]]],
                         dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    base = torch.linspace(-3.1415927, 3.1415927, steps=12, dtype=torch.float32).reshape(3, 4).numpy()
    input = base[:, ::2]
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([[-7.85398163, 7.85398163, 2.35619449],
                          [-2.35619449, 0.0, 4.71238898]], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.rad2deg"] = rad2deg_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.rad2deg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rad2deg'.")


check_valid('torch.rad2deg', generated_inputs['torch.rad2deg'], lib="torch", suffix=0)
