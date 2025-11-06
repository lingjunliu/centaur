
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def nanmedian_inputs_sig2():
    def out_shape(a, dim, keepdim):
        nd = a.ndim
        d = dim if dim >= 0 else dim + nd
        if keepdim:
            s = list(a.shape)
            s[d] = 1
            return tuple(s)
        else:
            return tuple(list(a.shape[:d]) + list(a.shape[d+1:]))

    list_of_inputs = []

    a1 = torch.tensor([1.0, float('nan'), 3.0, 2.0, float('nan')], dtype=torch.float32).numpy()
    dim, keepdim = 0, False
    s = out_shape(a1, dim, keepdim)
    out = (np.empty(s, dtype=a1.dtype), np.empty(s, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": a1, "dim": dim, "keepdim": keepdim, "out": out}))

    a2 = torch.tensor([[2.0, 3.0, 1.0], [float('nan'), 1.0, float('nan')]], dtype=torch.float64).numpy()
    dim, keepdim = 0, False
    s = out_shape(a2, dim, keepdim)
    out = (np.empty(s, dtype=a2.dtype), np.empty(s, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": a2, "dim": dim, "keepdim": keepdim, "out": out}))

    a3 = torch.tensor([[10.0, -5.0, 8.0], [7.0, 4.0, 9.0]], dtype=torch.float64).numpy()
    dim, keepdim = 1, True
    s = out_shape(a3, dim, keepdim)
    out = (np.empty(s, dtype=a3.dtype), np.empty(s, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": a3, "dim": dim, "keepdim": keepdim, "out": out}))

    a4 = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).numpy()
    a4[0, 1, 2] = np.nan
    a4[1, 2, 3] = np.nan
    dim, keepdim = -1, False
    s = out_shape(a4, dim, keepdim)
    out = (np.empty(s, dtype=a4.dtype), np.empty(s, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": a4, "dim": dim, "keepdim": keepdim, "out": out}))

    a5 = torch.tensor(
        [
            [[1.0, np.nan, 5.0, -1.0], [np.nan, np.nan, 0.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
            [[4.0, 2.0, 1.0, np.nan], [7.0, -8.0, 9.0, 10.0], [np.nan, np.nan, np.nan, np.nan]],
        ],
        dtype=torch.float32,
    ).numpy()
    dim, keepdim = 1, True
    s = out_shape(a5, dim, keepdim)
    out = (np.empty(s, dtype=a5.dtype), np.empty(s, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": a5, "dim": dim, "keepdim": keepdim, "out": out}))

    a6 = torch.randint(-10, 10, (4, 5), dtype=torch.int64).numpy()
    dim, keepdim = -2, False
    s = out_shape(a6, dim, keepdim)
    out = (np.empty(s, dtype=a6.dtype), np.empty(s, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": a6, "dim": dim, "keepdim": keepdim, "out": out}))

    a7 = torch.tensor([[[ -1,  2]], [[ 3, -4]], [[ 0,  5]]], dtype=torch.int32).numpy()
    dim, keepdim = 0, True
    s = out_shape(a7, dim, keepdim)
    out = (np.empty(s, dtype=a7.dtype), np.empty(s, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": a7, "dim": dim, "keepdim": keepdim, "out": out}))

    a8 = torch.tensor([[np.nan, 1.0, 2.0], [3.0, np.nan, 4.0], [5.0, 6.0, np.nan]], dtype=torch.float16).numpy()
    dim, keepdim = 1, False
    s = out_shape(a8, dim, keepdim)
    out = (np.empty(s, dtype=a8.dtype), np.empty(s, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": a8, "dim": dim, "keepdim": keepdim, "out": out}))

    a9 = torch.tensor(np.arange(16).reshape(2, 2, 2, 2), dtype=torch.float32).numpy()
    a9[0, 0, 1, 1] = np.nan
    a9[1, 1, 0, 0] = np.nan
    dim, keepdim = 2, True
    s = out_shape(a9, dim, keepdim)
    out = (np.empty(s, dtype=a9.dtype), np.empty(s, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": a9, "dim": dim, "keepdim": keepdim, "out": out}))

    a10 = torch.tensor(np.random.randn(1, 2, 1, 3, 2), dtype=torch.float64).numpy()
    a10[0, 1, 0, 2, 1] = np.nan
    dim, keepdim = -3, False
    s = out_shape(a10, dim, keepdim)
    out = (np.empty(s, dtype=a10.dtype), np.empty(s, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": a10, "dim": dim, "keepdim": keepdim, "out": out}))

    a11 = torch.tensor([[np.nan, np.nan, np.nan, np.nan], [1.0, -1.0, 0.0, np.nan]], dtype=torch.float32).numpy()
    dim, keepdim = 1, False
    s = out_shape(a11, dim, keepdim)
    out = (np.empty(s, dtype=a11.dtype), np.empty(s, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": a11, "dim": dim, "keepdim": keepdim, "out": out}))

    a12 = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32).numpy()
    dim, keepdim = 0, True
    s = out_shape(a12, dim, keepdim)
    out = (np.empty(s, dtype=a12.dtype), np.empty(s, dtype=np.int64))
    list_of_inputs.append(copy.deepcopy({"input": a12, "dim": dim, "keepdim": keepdim, "out": out}))

    return list_of_inputs

generated_inputs["torch.nanmedian_2"] = nanmedian_inputs_sig2()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nanmedian_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nanmedian_2'.")


check_valid('torch.nanmedian', generated_inputs['torch.nanmedian_2'], lib="torch", suffix=2)
