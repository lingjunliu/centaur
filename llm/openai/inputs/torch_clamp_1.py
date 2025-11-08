
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clamp_inputs():
    list_of_inputs = []
    
    input_arr = torch.tensor([-1.7120, 0.1734, -0.0478, -0.0922], dtype=torch.float32).numpy()
    min_val = -0.5
    max_val = 0.5
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_val, "out": out_arr}))
    
    input_arr = torch.randn((2, 3), dtype=torch.float64).numpy()
    min_val = -1.0
    max_val = 1.0
    out_arr = np.zeros_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_val, "out": out_arr}))
    
    input_arr = torch.linspace(-10, 10, steps=7, dtype=torch.float32).reshape(1, 7).numpy()
    min_val = -5.0
    max_val = 2.0
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_val, "out": out_arr}))
    
    input_arr = torch.tensor([[-10.0, 0.0, 10.0], [5.0, -5.0, 2.0]], dtype=torch.float32).numpy()
    min_val = -3.0
    max_val = 4.0
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_val, "out": out_arr}))
    
    input_arr = torch.randn((2, 2, 3), dtype=torch.float32).numpy()
    min_val = -0.1
    max_val = 0.1
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_val, "out": out_arr}))
    
    input_arr = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    min_val = -1.0
    max_val = 1.0
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_val, "out": out_arr}))
    
    input_arr = np.empty((0, 4), dtype=np.float32)
    min_val = -1.0
    max_val = 1.0
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_val, "out": out_arr}))
    
    input_arr = np.array([-1e20, 1e20, 3.14, -2.71], dtype=np.float64)
    min_val = -1e10
    max_val = 1e10
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_val, "out": out_arr}))
    
    input_arr = torch.arange(0, 24, dtype=torch.float32).reshape(1, 2, 3, 4).numpy()
    min_val = 5.0
    max_val = 15.0
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_val, "out": out_arr}))
    
    input_arr = np.linspace(-1, 1, 8, dtype=np.float32).reshape(2, 4)
    min_val = 2.0
    max_val = -2.0
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_val, "out": out_arr}))
    
    return list_of_inputs

generated_inputs["torch.clamp_1"] = clamp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.clamp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clamp_1'.")


check_valid('torch.clamp', generated_inputs['torch.clamp_1'], lib="torch", suffix=1)
