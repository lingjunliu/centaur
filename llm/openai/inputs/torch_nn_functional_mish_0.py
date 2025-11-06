
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def mish_inputs():
    list_of_inputs = []
    
    # 1
    input_arr = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 2
    input_arr = torch.tensor([[-10.0, -2.0, 0.5],
                              [1.2, 5.5, 20.0]], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 3
    input_arr = torch.linspace(-5, 5, steps=24, dtype=torch.float16).reshape(2, 3, 4).numpy()
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 4
    input_arr = torch.randn(1, 3, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 5
    input_arr = torch.tensor(3.14, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 6
    input_arr = torch.tensor([], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 7
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6).numpy()
    input_arr = base[:, ::2]
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 8
    input_arr = torch.tensor([100.0, -100.0, 50.0, -50.0], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 9
    input_arr = torch.randn(2, 1, 2, 1, 3, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 10
    input_arr = torch.randn(2, 3, 4, 5, dtype=torch.float16).numpy()
    input_dict = {
        "input": input_arr,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 11
    input_arr = np.asfortranarray(torch.randn(3, 3, dtype=torch.float32).numpy())
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 12
    input_arr = torch.tensor([float('nan'), float('inf'), float('-inf'), 0.0], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 13
    input_arr = torch.empty((2, 0, 3), dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.mish"] = mish_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.mish' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.mish'.")


check_valid('torch.nn.functional.mish', generated_inputs['torch.nn.functional.mish'], lib="torch", suffix=0)
