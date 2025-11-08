
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_min_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 with negatives and zeros
    input = np.array([1.0, -2.5, 0.0, 3.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 2: 2D int64 with negatives
    input = np.array([[-1, 0, 2], [5, -3, 4]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 3: 3D float64 with NaN and Inf
    input = np.array([[[np.nan, 2.0, -1.0],
                       [np.inf, 0.5, 3.0]],
                      [[-np.inf, -0.0, 0.0],
                       [1.0, 2.0, np.nan]]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 4: 0D scalar float64
    input = np.array(7.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 5: 1D boolean
    input = np.array([True, False, True, False], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 6: 2D float16
    input = np.array([[1.5, -2.0],
                      [3.25, -4.125]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 7: 3D non-contiguous int32 (via transpose)
    input = np.arange(24, dtype=np.int32).reshape(2, 3, 4).transpose(2, 1, 0)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 8: 2D float32 with -0.0 and 0.0
    input = np.array([[0.0, -0.0],
                      [-1.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 9: 4D float32 random
    rng = np.random.RandomState(0)
    input = rng.randn(2, 1, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 10: 1D int8 range extremes
    input = np.array([127, -128, 0, 1, -1], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 11: 1D uint8
    input = np.array([0, 255, 10, 200], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 12: 5D float32
    input = np.arange(1*2*1*2*3, dtype=np.float32).reshape(1, 2, 1, 2, 3) - 5.0
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.min_1"] = torch_min_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.min_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.min_1'.")


check_valid('torch.min', generated_inputs['torch.min_1'], lib="torch", suffix=1)
