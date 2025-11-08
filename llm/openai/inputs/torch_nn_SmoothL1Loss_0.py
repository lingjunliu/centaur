
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def smooth_l1loss_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.tensor([-1.0, 0.0, 1.5, -2.3, 3.3], dtype=torch.float32).numpy()
    target_arr = torch.tensor([0.5, -0.5, 1.0, -2.0, 3.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "beta": 1.0,
        "input": input_arr,
        "target": target_arr
    }))

    # 2
    input_arr = torch.ones((2, 3), dtype=torch.float32).numpy()
    target_arr = torch.zeros((2, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "beta": 0.5,
        "input": input_arr,
        "target": target_arr
    }))

    # 3
    input_arr = torch.randn(2, 2, 2, dtype=torch.float32).numpy()
    target_arr = torch.randn(2, 2, 2, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "beta": 1.5,
        "input": input_arr,
        "target": target_arr
    }))

    # 4
    input_arr = torch.tensor(3.0, dtype=torch.float32).numpy()
    target_arr = torch.tensor(-2.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "beta": 0.0,
        "input": input_arr,
        "target": target_arr
    }))

    # 5
    input_arr = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    target_arr = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "beta": 2.0,
        "input": input_arr,
        "target": target_arr
    }))

    # 6
    input_arr = torch.tensor([1e-3, -1e-3, 2.5, -2.5], dtype=torch.float64).numpy()
    target_arr = torch.tensor([0.0, 0.0, 2.0, -3.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "beta": 0.2,
        "input": input_arr,
        "target": target_arr
    }))

    # 7
    input_arr = torch.arange(9, dtype=torch.float16).reshape(3, 3).numpy()
    target_arr = torch.zeros((3, 3), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "beta": 1.0,
        "input": input_arr,
        "target": target_arr
    }))

    # 8
    input_arr = torch.tensor([-1000.0, 1000.0, -50.0, 50.0, 0.01, -0.01], dtype=torch.float32).numpy()
    target_arr = torch.zeros(6, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "beta": 0.1,
        "input": input_arr,
        "target": target_arr
    }))

    # 9
    input_arr = torch.randn(2, 1, 2, 1, 3, dtype=torch.float32).numpy()
    target_arr = torch.randn(2, 1, 2, 1, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "beta": 0.75,
        "input": input_arr,
        "target": target_arr
    }))

    # 10
    input_arr = torch.tensor([[[1.234]]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[[1.0]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "beta": 1e6,
        "input": input_arr,
        "target": target_arr
    }))

    # 11
    input_arr = torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "beta": 1e-8,
        "input": input_arr,
        "target": target_arr
    }))

    # 12
    input_arr = torch.tensor([[[2.0, -2.0, 0.5, -0.5, 1.0]]], dtype=torch.float32).expand(4, 1, 5).contiguous().numpy()
    target_arr = torch.tensor([[[1.5, -1.5, 0.0, 0.0, 1.5]]], dtype=torch.float32).expand(4, 1, 5).contiguous().numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "beta": 3.0,
        "input": input_arr,
        "target": target_arr
    }))

    return list_of_inputs

generated_inputs["torch.nn.SmoothL1Loss"] = smooth_l1loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.SmoothL1Loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.SmoothL1Loss'.")


check_valid('torch.nn.SmoothL1Loss', generated_inputs['torch.nn.SmoothL1Loss'], lib="torch", suffix=0)
