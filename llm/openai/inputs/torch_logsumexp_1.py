
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logsumexp_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([[0.5, -1.2, 3.0],
                          [2.1, 0.0, -0.7],
                          [-1.0, 4.0, 1.5]], dtype=np.float32)
    dim = 1
    keepdim = False
    out_shape = (input_arr.shape[0],)
    out = np.empty(out_shape, dtype=input_arr.dtype)
    input_dict = {"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = np.random.randn(2, 5).astype(np.float64)
    dim = 0
    keepdim = True
    out_shape = (1, input_arr.shape[1])
    out = np.empty(out_shape, dtype=input_arr.dtype)
    input_dict = {"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = np.random.randn(4, 3, 2).astype(np.float16)
    dim = -1
    keepdim = False
    dim_idx = dim + input_arr.ndim if dim < 0 else dim
    out_shape = input_arr.shape[:dim_idx] + input_arr.shape[dim_idx+1:]
    out = np.empty(out_shape, dtype=input_arr.dtype)
    input_dict = {"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = np.linspace(-3, 3, 6, dtype=np.float32)
    dim = 0
    keepdim = True
    out_shape = (1,)
    out = np.empty(out_shape, dtype=input_arr.dtype)
    input_dict = {"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = (np.random.randn(2, 3, 4, 5) * 2.0).astype(np.float32)
    dim = 2
    keepdim = False
    out_shape = (input_arr.shape[0], input_arr.shape[1], input_arr.shape[3])
    out = np.empty(out_shape, dtype=input_arr.dtype)
    input_dict = {"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = (np.random.randn(2, 4, 3) - 5.0).astype(np.float64)
    dim = 1
    keepdim = True
    out_shape = (input_arr.shape[0], 1, input_arr.shape[2])
    out = np.empty(out_shape, dtype=input_arr.dtype)
    input_dict = {"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = (np.random.randn(1, 5, 1, 7)).astype(np.float16)
    dim = -2
    keepdim = True
    dim_idx = dim + input_arr.ndim if dim < 0 else dim
    out_shape = list(input_arr.shape)
    out_shape[dim_idx] = 1
    out = np.empty(tuple(out_shape), dtype=input_arr.dtype)
    input_dict = {"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = np.random.uniform(-2, 2, size=(2, 2, 2, 2, 2)).astype(np.float32)
    dim = 3
    keepdim = False
    out_shape = input_arr.shape[:3] + input_arr.shape[4:]
    out = np.empty(out_shape, dtype=input_arr.dtype)
    input_dict = {"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = np.array([42.0], dtype=np.float64)
    dim = 0
    keepdim = False
    out_shape = ()
    out = np.empty(out_shape, dtype=input_arr.dtype)
    input_dict = {"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = np.array([1000.0, -1000.0, 0.0, 5.0, -5.0, 10.0, -10.0, 2.5, -2.5, 1.0], dtype=np.float32)
    dim = 0
    keepdim = False
    out_shape = ()
    out = np.empty(out_shape, dtype=input_arr.dtype)
    input_dict = {"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    input_arr = base.T  # shape (4, 3), non-contiguous view
    input_arr = np.ascontiguousarray(input_arr)  # ensure contiguous numpy array
    dim = -1
    keepdim = False
    dim_idx = dim + input_arr.ndim if dim < 0 else dim
    out_shape = input_arr.shape[:dim_idx] + input_arr.shape[dim_idx+1:]
    out = np.empty(out_shape, dtype=input_arr.dtype)
    input_dict = {"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = (np.random.randn(3, 3, 3) * 0.5 + 1.0).astype(np.float16)
    dim = 0
    keepdim = True
    out_shape = (1, input_arr.shape[1], input_arr.shape[2])
    out = np.empty(out_shape, dtype=input_arr.dtype)
    input_dict = {"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.logsumexp_1"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logsumexp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logsumexp_1'.")


check_valid('torch.logsumexp', generated_inputs['torch.logsumexp_1'], lib="torch", suffix=1)
