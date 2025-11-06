
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def median_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, dim=0, keepdim=False
    inp = np.array([1.5, -2.0, 3.3, 0.0, 4.2], dtype=np.float32)
    dim = 0
    keepdim = False
    out_shape = inp.shape[:dim] + inp.shape[dim+1:]
    out = (np.empty(out_shape, dtype=inp.dtype), np.empty(out_shape, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 2: 2D float64, dim=1, keepdim=False
    inp = (np.arange(20, dtype=np.float64) - 10.5).reshape(4, 5)
    dim = 1
    keepdim = False
    out_shape = inp.shape[:dim] + inp.shape[dim+1:]
    out = (np.empty(out_shape, dtype=inp.dtype), np.empty(out_shape, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 3: 2D int64, dim=0, keepdim=True
    inp = np.array([[5, 2, 7, 1],
                    [3, 8, 6, 4],
                    [9, 0, 5, 2]], dtype=np.int64)
    dim = 0
    keepdim = True
    out_shape = inp.shape[:dim] + (1,) + inp.shape[dim+1:]
    out = (np.empty(out_shape, dtype=inp.dtype), np.empty(out_shape, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 4: 3D float32, dim=-1, keepdim=False
    inp = np.linspace(-3, 3, num=2*3*5, dtype=np.float32).reshape(2, 3, 5)
    dim = -1
    keepdim = False
    axis = dim if dim >= 0 else inp.ndim + dim
    out_shape = inp.shape[:axis] + inp.shape[axis+1:]
    out = (np.empty(out_shape, dtype=inp.dtype), np.empty(out_shape, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 5: 3D int32, dim=0, keepdim=False
    inp = np.array([[[10, -1, 3, 7]],
                    [[-5, 4, 9, 2]],
                    [[0, -3, 8, 1]]], dtype=np.int32)
    dim = 0
    keepdim = False
    out_shape = inp.shape[:dim] + inp.shape[dim+1:]
    out = (np.empty(out_shape, dtype=inp.dtype), np.empty(out_shape, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 6: 4D float64, dim=2, keepdim=True
    inp = (np.arange(2*2*3*4, dtype=np.float64) / 10.0).reshape(2, 2, 3, 4)
    dim = 2
    keepdim = True
    out_shape = inp.shape[:dim] + (1,) + inp.shape[dim+1:]
    out = (np.empty(out_shape, dtype=inp.dtype), np.empty(out_shape, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 7: 5D float16, dim=-3, keepdim=False
    inp = np.arange(1*2*3*2*4, dtype=np.float16).reshape(1, 2, 3, 2, 4)
    dim = -3
    keepdim = False
    axis = dim if dim >= 0 else inp.ndim + dim
    out_shape = inp.shape[:axis] + inp.shape[axis+1:]
    out = (np.empty(out_shape, dtype=inp.dtype), np.empty(out_shape, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 8: 2D float32 with NaN/inf, dim=0, keepdim=True
    inp = np.array([[np.nan, 1.0, -np.inf, 3.5, 2.2, 0.0],
                    [5.1, np.inf, 4.0, -1.2, -3.3, 7.7]], dtype=np.float32)
    dim = 0
    keepdim = True
    out_shape = inp.shape[:dim] + (1,) + inp.shape[dim+1:]
    out = (np.empty(out_shape, dtype=inp.dtype), np.empty(out_shape, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 9: 3D float64, dim=1, keepdim=False
    inp = (np.arange(-24, -24 + 2*4*3, dtype=np.float64)).reshape(2, 4, 3)
    dim = 1
    keepdim = False
    out_shape = inp.shape[:dim] + inp.shape[dim+1:]
    out = (np.empty(out_shape, dtype=inp.dtype), np.empty(out_shape, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 10: 1D int64 even length, dim=0, keepdim=True
    inp = np.array([8, 2, 6, 4, 10, 0], dtype=np.int64)
    dim = 0
    keepdim = True
    out_shape = inp.shape[:dim] + (1,) + inp.shape[dim+1:]
    out = (np.empty(out_shape, dtype=inp.dtype), np.empty(out_shape, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 11: 4D int64, dim=-4, keepdim=False
    inp = np.arange(3*3*2*2, dtype=np.int64).reshape(3, 3, 2, 2) - 5
    dim = -4
    keepdim = False
    axis = dim if dim >= 0 else inp.ndim + dim
    out_shape = inp.shape[:axis] + inp.shape[axis+1:]
    out = (np.empty(out_shape, dtype=inp.dtype), np.empty(out_shape, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 12: 2D float64 with duplicate medians, dim=1, keepdim=True
    inp = np.array([[1.0, 1.0],
                    [2.0, 2.0],
                    [3.5, 3.5],
                    [0.0, 10.0],
                    [-5.0, -5.0],
                    [7.2, 7.2]], dtype=np.float64)
    dim = 1
    keepdim = True
    out_shape = inp.shape[:dim] + (1,) + inp.shape[dim+1:]
    out = (np.empty(out_shape, dtype=inp.dtype), np.empty(out_shape, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "keepdim": keepdim, "out": out}))

    return list_of_inputs

generated_inputs["torch.median_2"] = median_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.median_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.median_2'.")


check_valid('torch.median', generated_inputs['torch.median_2'], lib="torch", suffix=2)
