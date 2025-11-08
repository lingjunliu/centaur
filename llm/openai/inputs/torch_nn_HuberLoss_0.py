
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def huberloss_inputs():
    list_of_inputs = []

    # 1
    input_arr = np.array(0.5, dtype=np.float32)
    target_arr = np.array(1.0, dtype=np.float32)
    input_dict = {
        "reduction": "mean",
        "delta": 1.0,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = np.array([-1.5, 0.0, 2.5, -3.0, 4.5], dtype=np.float32)
    target_arr = np.array([0.5, -0.5, 2.0, -2.0, 5.0], dtype=np.float32)
    input_dict = {
        "reduction": "sum",
        "delta": 0.5,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = np.array([[1.0, -2.0, 3.0],
                          [4.5, -5.5, 6.5]], dtype=np.float64)
    target_arr = np.array([[1.5, -1.0, 2.0],
                           [4.0, -6.0, 6.0]], dtype=np.float64)
    input_dict = {
        "reduction": "none",
        "delta": 2.0,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input_arr = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    target_arr = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    input_dict = {
        "reduction": "mean",
        "delta": 1.5,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = torch.tensor([[[[1.0, -1.0], [2.0, -2.0]]],
                              [[[3.0, -3.0], [4.0, -4.0]]]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[[[0.5, -0.5], [1.5, -1.5]]],
                               [[[2.5, -2.5], [3.5, -3.5]]]], dtype=torch.float32).numpy()
    input_dict = {
        "reduction": "sum",
        "delta": 0.1,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = np.linspace(-10, 10, num=3).astype(np.float64)
    target_arr = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    input_dict = {
        "reduction": "none",
        "delta": 10.0,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    rng1 = np.random.RandomState(0)
    rng2 = np.random.RandomState(1)
    input_arr = rng1.randn(4, 4).astype(np.float32)
    target_arr = rng2.randn(4, 4).astype(np.float32)
    input_dict = {
        "reduction": "mean",
        "delta": 100.0,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input_arr = np.array([1e-3, -1e-3, 2e-3, -2e-3, 0.0, 5e-3], dtype=np.float32)
    target_arr = np.zeros(6, dtype=np.float32)
    input_dict = {
        "reduction": "sum",
        "delta": 1e-6,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = np.array([[[-1.0], [0.5], [3.0]],
                          [[-2.5], [2.0], [0.0]]], dtype=np.float64)
    target_arr = np.array([[[-1.5], [0.0], [2.5]],
                           [[-3.0], [1.5], [-0.5]]], dtype=np.float64)
    input_dict = {
        "reduction": "none",
        "delta": 0.75,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    input_arr = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float16)
    target_arr = np.array([[0.5, -0.5], [1.5, -1.5]], dtype=np.float16)
    input_dict = {
        "reduction": "mean",
        "delta": 1.0,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = torch.linspace(-1, 1, steps=12, dtype=torch.float32).reshape(3, 4).numpy()
    target_arr = torch.zeros(3, 4, dtype=torch.float32).numpy()
    input_dict = {
        "reduction": "sum",
        "delta": 0.25,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    input_arr = np.array(-3.5, dtype=np.float64)
    target_arr = np.array(0.0, dtype=np.float64)
    input_dict = {
        "reduction": "sum",
        "delta": 3.0,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.HuberLoss"] = huberloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.HuberLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.HuberLoss'.")


check_valid('torch.nn.HuberLoss', generated_inputs['torch.nn.HuberLoss'], lib="torch", suffix=0)
