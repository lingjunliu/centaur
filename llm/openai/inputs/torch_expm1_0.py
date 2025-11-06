
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def expm1_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    input_arr = np.array([-10.0, -1.0, 0.0, 0.5, 1.0], dtype=np.float32)
    out_arr = np.empty_like(input_arr)
    input_dict = {"input": input_arr, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64
    input_arr = np.linspace(-5, 5, 6, dtype=np.float64).reshape(2, 3)
    out_arr = np.zeros_like(input_arr)
    input_dict = {"input": input_arr, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: scalar float32
    input_arr = np.array(0.0, dtype=np.float32)
    out_arr = np.array(0.0, dtype=np.float32)
    input_dict = {"input": input_arr, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float16
    input_arr = np.array(
        [
            [[-0.1, 0.2, -0.3, 0.4],
             [0.5, -0.6, 0.7, -0.8],
             [0.9, -1.0, 1.1, -1.2]],
            [[-1.3, 1.4, -1.5, 1.6],
             [-1.7, 1.8, -1.9, 2.0],
             [-2.1, 2.2, -2.3, 2.4]]
        ],
        dtype=np.float16
    )
    out_arr = np.zeros_like(input_arr)
    input_dict = {"input": input_arr, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D complex64
    input_arr = np.array([1+1j, -1+2j, 3-3j, -4-0.5j], dtype=np.complex64)
    out_arr = np.empty_like(input_arr)
    input_dict = {"input": input_arr, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex128
    input_arr = np.array([[0+0j, -2+0j],
                          [0.5-0.1j, -0.5+0.1j]], dtype=np.complex128)
    out_arr = np.zeros_like(input_arr)
    input_dict = {"input": input_arr, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large magnitudes float64
    input_arr = np.array([50.0, 100.0, -100.0], dtype=np.float64)
    out_arr = np.empty_like(input_arr)
    input_dict = {"input": input_arr, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small magnitudes float64
    input_arr = np.array([1e-12, -1e-12, 1e-8, -1e-8], dtype=np.float64)
    out_arr = np.zeros_like(input_arr)
    input_dict = {"input": input_arr, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-contiguous slice float32
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    input_arr = (base[:, ::2] * 0.1)
    out_arr = np.empty_like(input_arr)
    input_dict = {"input": input_arr, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty array float32 (0, 3)
    input_arr = np.empty((0, 3), dtype=np.float32)
    out_arr = np.empty_like(input_arr)
    input_dict = {"input": input_arr, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D float32 with NaN and Inf
    input_arr = np.array([[[[0.0, np.nan], [np.inf, -np.inf]]]], dtype=np.float32)
    out_arr = np.zeros_like(input_arr)
    input_dict = {"input": input_arr, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Column vector float32
    input_arr = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float32)
    out_arr = np.zeros_like(input_arr)
    input_dict = {"input": input_arr, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.expm1"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.expm1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.expm1'.")


check_valid('torch.expm1', generated_inputs['torch.expm1'], lib="torch", suffix=0)
