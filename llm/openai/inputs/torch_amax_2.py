
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def amax_inputs():
    def out_shape(shape, dims, keepdim):
        ndim = len(shape)
        pos = [(d if d >= 0 else d + ndim) for d in dims]
        if keepdim:
            return tuple(1 if i in pos else shape[i] for i in range(ndim))
        else:
            return tuple(shape[i] for i in range(ndim) if i not in pos)

    def make_out(shape, dtype):
        return torch.empty(shape, dtype=dtype).numpy()

    list_of_inputs = []

    # Input 1: 2D float32, reduce dim 1
    t = torch.randn(4, 4, dtype=torch.float32)
    dims = (1,)
    keepdim = False
    input_np = t.numpy()
    out_np = make_out(out_shape(input_np.shape, dims, keepdim), t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_np, "dim": dims, "keepdim": keepdim, "out": out_np}))

    # Input 2: 2D float32, reduce dim 0, keepdim True
    t = (torch.randn(3, 5, dtype=torch.float32) * 10 - 5)
    dims = (0,)
    keepdim = True
    input_np = t.numpy()
    out_np = make_out(out_shape(input_np.shape, dims, keepdim), t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_np, "dim": dims, "keepdim": keepdim, "out": out_np}))

    # Input 3: 3D float64, reduce dims (1,2)
    t = torch.randn(2, 3, 4, dtype=torch.float64)
    dims = (1, 2)
    keepdim = False
    input_np = t.numpy()
    out_np = make_out(out_shape(input_np.shape, dims, keepdim), t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_np, "dim": dims, "keepdim": keepdim, "out": out_np}))

    # Input 4: 3D float64, reduce last dim with keepdim True
    t = torch.randn(2, 3, 4, dtype=torch.float64)
    dims = (-1,)
    keepdim = True
    input_np = t.numpy()
    out_np = make_out(out_shape(input_np.shape, dims, keepdim), t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_np, "dim": dims, "keepdim": keepdim, "out": out_np}))

    # Input 5: 4D int64, reduce dims (2,3) with keepdim True
    t = torch.randint(-100, 100, (2, 3, 4, 5), dtype=torch.int64)
    dims = (2, 3)
    keepdim = True
    input_np = t.numpy()
    out_np = make_out(out_shape(input_np.shape, dims, keepdim), t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_np, "dim": dims, "keepdim": keepdim, "out": out_np}))

    # Input 6: 4D int32, reduce dim 0
    t = torch.randint(-50, 50, (2, 3, 4, 5), dtype=torch.int32)
    dims = (0,)
    keepdim = False
    input_np = t.numpy()
    out_np = make_out(out_shape(input_np.shape, dims, keepdim), t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_np, "dim": dims, "keepdim": keepdim, "out": out_np}))

    # Input 7: 1D bool, reduce dim 0
    t = (torch.rand(7) > 0.4)
    dims = (0,)
    keepdim = False
    input_np = t.numpy()
    out_np = make_out(out_shape(input_np.shape, dims, keepdim), torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_np, "dim": dims, "keepdim": keepdim, "out": out_np}))

    # Input 8: 5D float16, reduce dims (1,3) with keepdim True
    t = torch.randn(2, 2, 3, 1, 4, dtype=torch.float16)
    dims = (1, 3)
    keepdim = True
    input_np = t.numpy()
    out_np = make_out(out_shape(input_np.shape, dims, keepdim), t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_np, "dim": dims, "keepdim": keepdim, "out": out_np}))

    # Input 9: 3D float32 with NaNs, reduce dims (0,2)
    t = torch.randn(5, 6, 7, dtype=torch.float32)
    t[0, 0, 0] = torch.nan
    t[2, 3, 4] = torch.nan
    dims = (0, 2)
    keepdim = False
    input_np = t.numpy()
    out_np = make_out(out_shape(input_np.shape, dims, keepdim), t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_np, "dim": dims, "keepdim": keepdim, "out": out_np}))

    # Input 10: 2D float32, reduce dims (-2,-1) keepdim True
    t = torch.randn(4, 5, dtype=torch.float32)
    dims = (-2, -1)
    keepdim = True
    input_np = t.numpy()
    out_np = make_out(out_shape(input_np.shape, dims, keepdim), t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_np, "dim": dims, "keepdim": keepdim, "out": out_np}))

    # Input 11: 3D float32, full reduction to scalar
    t = torch.randn(2, 2, 2, dtype=torch.float32)
    dims = (0, 1, 2)
    keepdim = False
    input_np = t.numpy()
    out_np = make_out(out_shape(input_np.shape, dims, keepdim), t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_np, "dim": dims, "keepdim": keepdim, "out": out_np}))

    # Input 12: 6D int16, reduce dims (1,3,5)
    t = torch.randint(-200, 200, (3, 1, 4, 1, 2, 5), dtype=torch.int16)
    dims = (1, 3, 5)
    keepdim = False
    input_np = t.numpy()
    out_np = make_out(out_shape(input_np.shape, dims, keepdim), t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_np, "dim": dims, "keepdim": keepdim, "out": out_np}))

    return list_of_inputs

generated_inputs["torch.amax_2"] = amax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.amax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.amax_2'.")


check_valid('torch.amax', generated_inputs['torch.amax_2'], lib="torch", suffix=2)
