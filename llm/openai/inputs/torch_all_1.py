
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def torch_all_inputs():
    list_of_inputs = []
    
    input = torch.tensor([[False, True], [True, True]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.arange(0, 5, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([-1, 2, 0, 4], dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.ones((3, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.zeros((2, 2, 2), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor(0.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.arange(12, dtype=torch.int64).view(3, 4).t().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([float('nan'), 1.0, -0.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([0, 0, 0], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([0+0j, 1+0j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.empty((2, 0, 3), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.ones((1, 1, 1, 1), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    return list_of_inputs

generated_inputs["torch.all_1"] = torch_all_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.all_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.all_1'.")


check_valid('torch.all', generated_inputs['torch.all_1'], lib="torch", suffix=1)
