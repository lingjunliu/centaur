
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def round_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([4.7, -2.3, 9.1, -7.7], dtype=torch.float32).numpy()
    decimals = 0
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "decimals": decimals, "out": out}))

    input_arr = torch.tensor([-0.5, 0.5, 1.5, 2.5], dtype=torch.float64).numpy()
    decimals = 0
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "decimals": decimals, "out": out}))

    input_arr = torch.tensor([0.1234567], dtype=torch.float64).numpy()
    decimals = 3
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "decimals": decimals, "out": out}))

    input_arr = torch.tensor([1200.1234567], dtype=torch.float64).numpy()
    decimals = -3
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "decimals": decimals, "out": out}))

    input_arr = torch.tensor([[1.2345, -2.3456], [100.125, -100.875]], dtype=torch.float32).numpy()
    decimals = 2
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "decimals": decimals, "out": out}))

    input_arr = torch.linspace(-3.0, 3.0, steps=12, dtype=torch.float32).reshape(2, 2, 3).numpy()
    decimals = 1
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "decimals": decimals, "out": out}))

    input_arr = torch.tensor(2.5, dtype=torch.float64).numpy()
    decimals = 0
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "decimals": decimals, "out": out}))

    input_arr = torch.tensor([float('nan'), float('inf'), float('-inf'), 1.4999, 1.5], dtype=torch.float32).numpy()
    decimals = 0
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "decimals": decimals, "out": out}))

    input_arr = torch.tensor([1500.0, 2499.99, -3500.0], dtype=torch.float64).numpy()
    decimals = -2
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "decimals": decimals, "out": out}))

    input_arr = torch.tensor([[12.34, -25.55, 999.49], [0.05, -1.15, 5.95]], dtype=torch.float32).numpy()
    decimals = -1
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "decimals": decimals, "out": out}))

    base = torch.arange(10, dtype=torch.float32).reshape(2, 5).numpy().T.copy()
    input_arr = base
    decimals = 0
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "decimals": decimals, "out": out}))

    return list_of_inputs

generated_inputs["torch.round"] = round_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.round'.")


check_valid('torch.round', generated_inputs['torch.round'], lib="torch", suffix=0)
