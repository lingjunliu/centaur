
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, dtype=None
    input = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    dim = 0
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "dtype": dtype}))

    # Input 2: 2D int64 -> cast to torch.float64
    input = np.array([[1, -2, 3],
                      [-4, 5, -6]], dtype=np.int64)
    dim = 1
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "dtype": dtype}))

    # Input 3: 2D float32 along dim 0
    input = (np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0)
    dim = 0
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "dtype": dtype}))

    # Input 4: 3D float32 along last dim, cast to torch.float64
    input = (np.arange(2*3*4, dtype=np.float32).reshape(2, 3, 4) - 6.0) / 3.0
    dim = 2
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "dtype": dtype}))

    # Input 5: 3D int32 with negatives along dim -1 -> cast to torch.float32
    input = np.array([[[1, -1, 2, -2, 3]],
                      [[-3, 4, -4, 5, -5]],
                      [[6, -6, 7, -7, 8]],
                      [[-8, 9, -9, 10, -10]]], dtype=np.int32)
    dim = -1
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "dtype": dtype}))

    # Input 6: 4D float32 along dim 1, dtype=None
    input = np.linspace(-3, 3, num=2*3*4*5, dtype=np.float32).reshape(2, 3, 4, 5)
    dim = 1
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "dtype": dtype}))

    # Input 7: 4D float64 non-contiguous via transpose along dim -2
    base = np.arange(2*3*4*5, dtype=np.float64).reshape(2, 3, 4, 5)
    input = np.transpose(base, (0, 2, 1, 3))  # shape (2,4,3,5)
    dim = -2
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "dtype": dtype}))

    # Input 8: 5D int16 cast to torch.float32 along dim 3
    input = np.arange(2*2*2*2*2, dtype=np.int16).reshape(2, 2, 2, 2, 2) - 8
    dim = 3
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "dtype": dtype}))

    # Input 9: 2D with extreme values, float64 along last dim, dtype=None
    input = np.array([[1000.0, -1000.0, 0.0, 5.0],
                      [-1e9, 1e9, 1.0, -1.0],
                      [50.0, 50.0, 50.0, 50.0]], dtype=np.float64)
    dim = -1
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "dtype": dtype}))

    # Input 10: 1D int32 logits cast to torch.float32
    input = np.array([10, -10, 0, 3, -3, 7], dtype=np.int32)
    dim = 0
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "dtype": dtype}))

    # Input 11: 3D float64 along dim 0
    input = (np.random.RandomState(0).randn(4, 3, 2)).astype(np.float64)
    dim = 0
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "dtype": dtype}))

    # Input 12: 2D float32 with singleton batch along dim 1, dtype=None
    input = np.array([[0.5, -0.5, 2.0, -2.0, 0.0]], dtype=np.float32)
    dim = 1
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "dtype": dtype}))

    return list_of_inputs

generated_inputs["torch.nn.functional.softmax"] = softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.softmax'.")


check_valid('torch.nn.functional.softmax', generated_inputs['torch.nn.functional.softmax'], lib="torch", suffix=0)
