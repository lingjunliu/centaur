
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def l1loss_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, reduction='mean'
    input_arr = torch.randn(5, dtype=torch.float32).numpy()
    target_arr = torch.randn(5, dtype=torch.float32).numpy()
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32, reduction='sum'
    input_arr = torch.randn(3, 4, dtype=torch.float32).numpy()
    target_arr = torch.randn(3, 4, dtype=torch.float32).numpy()
    input_dict = {
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32, reduction='none'
    input_arr = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    target_arr = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    input_dict = {
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar float64, reduction='mean'
    input_arr = torch.tensor(3.14, dtype=torch.float64).numpy()
    target_arr = torch.tensor(2.71, dtype=torch.float64).numpy()
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 with larger magnitude values, reduction='sum'
    input_arr = (10 * torch.randn(2, 1, 3, 3, dtype=torch.float32)).numpy()
    target_arr = (10 * torch.randn(2, 1, 3, 3, dtype=torch.float32)).numpy()
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 with large constants, reduction='mean'
    input_arr = torch.full((2, 2), 1e6, dtype=torch.float32).numpy()
    target_arr = (torch.full((2, 2), 1e6, dtype=torch.float32) + 100.0 * torch.randn(2, 2)).numpy()
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D complex, reduction='mean'
    input_arr = (torch.randn(6) + 1j * torch.randn(6)).numpy()
    target_arr = (torch.randn(6) + 1j * torch.randn(6)).numpy()
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D complex, reduction='sum'
    input_arr = (torch.randn(2, 3) + 1j * torch.randn(2, 3)).numpy()
    target_arr = (torch.randn(2, 3) + 1j * torch.randn(2, 3)).numpy()
    input_dict = {
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float32, reduction='none'
    input_arr = torch.randn(2, 2, 2, 2, 2, dtype=torch.float32).numpy()
    target_arr = torch.randn(2, 2, 2, 2, 2, dtype=torch.float32).numpy()
    input_dict = {
        "size_average": False,
        "reduce": False,
        "reduction": "none",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D zeros vs random, reduction='mean'
    input_arr = torch.zeros(3, 5, dtype=torch.float32).numpy()
    target_arr = torch.randn(3, 5, dtype=torch.float32).numpy()
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D float64 with both positive and negative values, reduction='sum'
    input_arr = torch.linspace(-5.0, 5.0, steps=11, dtype=torch.float64).numpy()
    target_arr = (torch.linspace(5.0, -5.0, steps=11, dtype=torch.float64) + 0.5).numpy()
    input_dict = {
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 3D float16, reduction='none'
    input_arr = torch.randn(4, 2, 3, dtype=torch.float16).numpy()
    target_arr = torch.randn(4, 2, 3, dtype=torch.float16).numpy()
    input_dict = {
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.L1Loss"] = l1loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.L1Loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.L1Loss'.")


check_valid('torch.nn.L1Loss', generated_inputs['torch.nn.L1Loss'], lib="torch", suffix=0)
