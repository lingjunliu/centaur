
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def adaptive_max_pool1d_inputs():
    list_of_inputs = []
    
    # Input 1
    input_arr = torch.randn(1, 64, 8, dtype=torch.float32).numpy()
    input_dict = {
        "output_size": (5,),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_arr = torch.randn(64, 8, dtype=torch.float64).numpy()
    input_dict = {
        "output_size": (1,),
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_arr = (torch.randn(2, 3, 10, dtype=torch.float32) * 10 - 5).numpy()
    input_dict = {
        "output_size": (10,),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_arr = torch.linspace(-3.0, 3.0, steps=28, dtype=torch.float64).reshape(4, 1, 7).numpy()
    input_dict = {
        "output_size": (3,),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_arr = (torch.arange(15, dtype=torch.float32).reshape(1, 15) - 7.0).numpy()
    input_dict = {
        "output_size": (5,),
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_arr = torch.ones(3, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "output_size": (1,),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_arr = (torch.randn(8, 8, dtype=torch.float32) * 0.1).numpy()
    input_dict = {
        "output_size": (8,),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_arr = torch.randn(2, 2, 9, dtype=torch.float16).numpy()
    input_dict = {
        "output_size": (4,),
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 (non-contiguous slice with positive stride)
    base = torch.randn(2, 4, 12, dtype=torch.float32)
    sliced = base[:, :, ::2]  # shape (2, 4, 6)
    input_arr = sliced.numpy()
    input_dict = {
        "output_size": (3,),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_arr = (torch.tensor([[[-1.0, 2.0, -3.0]],
                               [[4.0, -5.0, 6.0]],
                               [[-7.0, 8.0, -9.0]]], dtype=torch.float32).squeeze(1)).numpy()  # shape (3,3)
    input_dict = {
        "output_size": (2,),
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_arr = torch.randn(5, 10, 20, dtype=torch.float32).numpy()
    input_dict = {
        "output_size": (5,),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    input_arr = torch.tensor([[0.5]], dtype=torch.float32).repeat(2, 1).numpy()  # shape (2,1)
    input_dict = {
        "output_size": (1,),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.AdaptiveMaxPool1d_2"] = adaptive_max_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AdaptiveMaxPool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveMaxPool1d_2'.")


check_valid('torch.nn.AdaptiveMaxPool1d', generated_inputs['torch.nn.AdaptiveMaxPool1d_2'], lib="torch", suffix=2)
