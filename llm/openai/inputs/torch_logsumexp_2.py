
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logsumexp_inputs():
    def compute_out(arr, dims, keepdim):
        nd = arr.ndim
        dims_pos = tuple(d if d >= 0 else nd + d for d in dims)
        shape = []
        for i in range(nd):
            if i in dims_pos:
                if keepdim:
                    shape.append(1)
            else:
                shape.append(arr.shape[i])
        return np.empty(tuple(shape), dtype=arr.dtype)

    list_of_inputs = []

    # 1
    input_arr = torch.randn(3, 3, dtype=torch.float32).numpy()
    dim = (1,)
    keepdim = False
    out = compute_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 2
    input_arr = torch.randn(3, 3, dtype=torch.float64).numpy()
    dim = (0,)
    keepdim = True
    out = compute_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 3
    input_arr = torch.tensor([[-1.0, -2.5, 0.0, 3.0, -4.0]], dtype=torch.float16).reshape(-1).numpy()
    dim = (0,)
    keepdim = False
    out = compute_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 4
    input_arr = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    dim = (2,)
    keepdim = False
    out = compute_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 5
    input_arr = torch.randn(2, 3, 4, dtype=torch.float64).numpy()
    dim = (-1,)
    keepdim = True
    out = compute_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 6
    input_arr = torch.randn(2, 0, 3, 5, dtype=torch.float32).numpy()
    dim = (1,)
    keepdim = False
    out = compute_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 7
    input_arr = torch.randn(4, 4, 4, dtype=torch.float32).numpy()
    dim = (0, 2)
    keepdim = False
    out = compute_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 8
    input_arr = torch.randn(1, 2, 1, 3, 2, dtype=torch.float16).numpy()
    dim = (2, 4)
    keepdim = True
    out = compute_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 9
    input_arr = np.array([[1000.0, -1000.0, 500.0, -500.0],
                          [1e20, -1e20, 1e10, -1e10]], dtype=np.float64)
    dim = (1,)
    keepdim = False
    out = compute_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 10
    input_arr = torch.randn(3, 5, 7, dtype=torch.float32).numpy()
    dim = (0, 1)
    keepdim = False
    out = compute_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 11
    input_arr = torch.randn(3, 2, 2, 2, dtype=torch.float64).numpy()
    dim = (-2,)
    keepdim = True
    out = compute_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 12
    input_arr = torch.randn(0, 0, dtype=torch.float32).numpy()
    dim = (1,)
    keepdim = False
    out = compute_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    return list_of_inputs

generated_inputs["torch.logsumexp_2"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logsumexp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logsumexp_2'.")


check_valid('torch.logsumexp', generated_inputs['torch.logsumexp_2'], lib="torch", suffix=2)
