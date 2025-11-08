
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def digamma_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 positive values
    input_arr = torch.tensor([0.1, 1.0, 2.5, 10.0], dtype=torch.float32).numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out}))

    # Input 2: 2D float64 values
    input_arr = (torch.arange(1, 7, dtype=torch.float64).reshape(2, 3) * 0.3).numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out}))

    # Input 3: 1D float64 with negative non-integers and positive
    input_arr = torch.tensor([-0.5, -1.3, -2.7, 3.5], dtype=torch.float64).numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out}))

    # Input 4: 0-D scalar float32
    input_arr = torch.tensor(3.14159265, dtype=torch.float32).numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out}))

    # Input 5: 3D float16 positive values
    input_arr = (torch.rand((2, 2, 3), dtype=torch.float16) + 0.1).numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out}))

    # Input 6: 1D float64 large/small magnitudes
    input_arr = torch.tensor([1000.0, 1e-6, 50.0, 0.5], dtype=torch.float64).numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out}))

    # Input 7: 1D float64 near poles (but not integers)
    input_arr = torch.tensor([-1.0001, -2.0001, -0.9999, -3.00001], dtype=torch.float64).numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out}))

    # Input 8: 1D float64 non-contiguous slice
    t = torch.linspace(0.1, 5.0, steps=11, dtype=torch.float64)
    input_arr = t[::2].numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out}))

    # Input 9: 2D float32 mixed signs (avoid nonpositive integers)
    input_arr = torch.tensor([[-0.25, 0.25, 1.25],
                              [2.75, -3.5, 4.0]], dtype=torch.float32).numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out}))

    # Input 10: 4D float64 positive values
    input_arr = (torch.rand((1, 3, 2, 2), dtype=torch.float64) + 0.01).numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out}))

    # Input 11: 1D float64 large negative non-integers
    input_arr = torch.tensor([-100.2, -50.5, -10.1, -0.1], dtype=torch.float64).numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out}))

    # Input 12: 1D float64 very small positives
    input_arr = torch.tensor([1e-8, 1e-6, 1e-4, 1e-2], dtype=torch.float64).numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.digamma"] = digamma_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.digamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.digamma'.")


check_valid('torch.special.digamma', generated_inputs['torch.special.digamma'], lib="torch", suffix=0)
