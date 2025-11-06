
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def nanmedian_inputs():
    list_of_inputs = []
    
    input = torch.tensor([1.0, float('nan'), 3.0, 2.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[2.0, 3.0, 1.0], [float('nan'), 1.0, float('nan')]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[[1.0, float('nan')], [3.5, -2.0]],
                          [[float('nan'), float('nan')], [0.0, 7.0]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([float('nan'), float('nan'), float('nan')], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor(5.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([-5, 0, 7, 2], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[-float('inf'), float('nan'), 2.0],
                          [3.0, 4.0, float('inf')]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[[0, 255], [10, 20]],
                          [[30, 40], [50, 60]]], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[[-1.0, -2.0], [float('nan'), 5.0]],
                          [[10.0, float('nan')], [3.0, 4.0]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.arange(16, dtype=torch.int32).reshape(2, 2, 2, 2).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([float('nan'), -100.0, float('nan'), 0.0, 100.0, float('nan')], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([1e-9, 1e9, -1e9, float('nan'), 5.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    return list_of_inputs

generated_inputs["torch.nanmedian_1"] = nanmedian_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nanmedian_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nanmedian_1'.")


check_valid('torch.nanmedian', generated_inputs['torch.nanmedian_1'], lib="torch", suffix=1)
