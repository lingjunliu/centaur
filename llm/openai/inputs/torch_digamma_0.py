
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def digamma_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 positives
    input = torch.tensor([0.1, 1.0, 2.5, 5.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 2: 2D float64 mixed non-integers
    input = torch.tensor([[-0.5, 0.5, 1.5],
                          [2.1, -3.7, 4.0]], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 3: 3D float32 small and mixed values
    input = torch.tensor([[[1e-6, 1e-3],
                           [0.1, 0.2]],
                          [[-0.9, -1.2],
                           [3.3, 7.7]]], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 4: 0-d scalar float64
    input = torch.tensor(3.141592653589793, dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 5: 4D float32 negatives (non-integers) and positives
    input = torch.tensor([[[[-2.5, -1.5, -0.1]],
                           [[0.3, 1.2, 2.8]]]], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 6: 1D float32 large values
    input = torch.tensor([10.0, 100.0, 1000.0, 1e6], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 7: Empty 2D float64
    input = torch.zeros((0, 3), dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 8: 1D float64 values near poles
    input = torch.tensor([-1.0 + 1e-6, -2.0 + 1e-6, -10.0 + 1e-8, 0.0 + 1e-12], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 9: 2D float32 including zeros and positives
    input = torch.tensor([[0.0, 1.0, 2.0],
                          [3.0, 4.5, 6.75]], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 10: 1D float64 infinities and NaN
    input = torch.tensor([float('inf'), float('nan'), -float('inf'), 1.0], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 11: 5D empty float32 (zero-size dimension)
    input = torch.empty((1, 0, 2, 3, 4), dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.digamma"] = digamma_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.digamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.digamma'.")


check_valid('torch.digamma', generated_inputs['torch.digamma'], lib="torch", suffix=0)
