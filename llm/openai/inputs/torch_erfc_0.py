
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def erfc_inputs():
    list_of_inputs = []

    # Input 1: scalar float32
    input = np.array(0.0, dtype=np.float32)
    out = np.empty((), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 2: 1D float32 with negatives and positives
    input = np.array([-3.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 3: 2D float64
    input = np.array([[0.1, -0.2, 3.3], [4.5, -5.6, 0.0]], dtype=np.float64)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 4: 3D float32
    input = np.linspace(-2.0, 2.0, num=24, dtype=np.float32).reshape(2, 3, 4)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 5: empty 1D float64
    input = np.array([], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 6: special values float64
    input = np.array([np.inf, -np.inf, np.nan, -0.0, 0.0, 10.0, -10.0], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 7: 4D float32 with zero-sized dim
    input = np.empty((2, 0, 3, 4), dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 8: 5D float32 small shape
    input = np.arange(6, dtype=np.float32).reshape(1, 2, 1, 3, 1)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 9: 2D float16
    input = np.array([[1.0, -1.0, 0.5], [-0.5, 2.0, -2.0]], dtype=np.float16)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 10: large 1D float64 vector
    input = np.random.default_rng(0).standard_normal(1000).astype(np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 11: 2D float32 with varied magnitudes
    input = np.array([[1e-6, -1e-3, 1e-1], [1.0, -5.0, 20.0], [-20.0, 0.0, 3.14159]], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 12: non-contiguous 2D float64 (transpose)
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    input = base.T
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.erfc"] = erfc_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.erfc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.erfc'.")


check_valid('torch.erfc', generated_inputs['torch.erfc'], lib="torch", suffix=0)
