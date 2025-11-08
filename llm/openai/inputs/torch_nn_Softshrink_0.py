
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def softshrink_inputs():
    list_of_inputs = []

    # 1: 1D float32 random values
    inp = np.random.randn(5).astype(np.float32)
    lambd = 0.5
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    # 2: 2D float32 with negatives and positives, lambd=0.0
    inp = np.linspace(-2.0, 2.0, num=6, dtype=np.float32).reshape(2, 3)
    lambd = 0.0
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    # 3: 0D scalar float32
    inp = np.array(3.0, dtype=np.float32)
    lambd = 1.0
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    # 4: 3D float32 normal distribution
    inp = (np.random.randn(2, 2, 2)).astype(np.float32)
    lambd = 0.2
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    # 5: 4D float32 (NCHW-like)
    inp = (np.random.randn(1, 3, 4, 4)).astype(np.float32)
    lambd = 0.7
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    # 6: 1D float64 with large magnitudes
    inp = np.array([-10.0, 0.0, 10.0], dtype=np.float64)
    lambd = 5.0
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    # 7: 1D float16 values
    inp = np.array([ -1.5, -0.2, 0.0, 0.2, 1.5, 3.0 ], dtype=np.float16)
    lambd = 0.3
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    # 8: 2D Fortran-ordered array
    inp = np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3) - 2.5)
    lambd = 0.5
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    # 9: 1D non-contiguous sliced array
    inp = np.arange(-5, 6, dtype=np.float32)[::2]
    lambd = 0.4
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    # 10: Empty array
    inp = np.array([], dtype=np.float32)
    lambd = 0.1
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    # 11: 5D array with broadcasting-friendly singleton dims
    inp = (np.random.randn(2, 1, 3, 1, 4)).astype(np.float32)
    lambd = 1.5
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    # 12: Boundary values around lambda
    inp = np.array([0.5, -0.5, 0.49, -0.49, 1.0, -1.0], dtype=np.float32)
    lambd = 0.5
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    # 13: Very small lambda
    inp = (np.random.randn(4, 4)).astype(np.float32)
    lambd = 1e-8
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    # 14: Very large lambda
    inp = np.array([100.0, -50.0, 0.1, -0.1], dtype=np.float32)
    lambd = 100.0
    list_of_inputs.append(copy.deepcopy({"lambd": float(lambd), "input": inp}))

    return list_of_inputs

generated_inputs["torch.nn.Softshrink"] = softshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Softshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softshrink'.")


check_valid('torch.nn.Softshrink', generated_inputs['torch.nn.Softshrink'], lib="torch", suffix=0)
