
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def torch_max_2_inputs():
    def make_out(arr, dim, keepdim):
        ndim = arr.ndim
        d = dim if dim >= 0 else dim + ndim
        if keepdim:
            shape = list(arr.shape)
            shape[d] = 1
        else:
            shape = list(arr.shape)
            shape.pop(d)
        out_shape = tuple(shape)
        values = np.empty(out_shape, dtype=arr.dtype)
        indices = np.empty(out_shape, dtype=np.int64)
        return (values, indices)

    list_of_inputs = []

    # 1
    input_arr = torch.tensor([0.5, -1.2, 3.3, 3.3], dtype=torch.float32).numpy()
    dim = 0
    keepdim = False
    out = make_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 2
    input_arr = torch.tensor([[1.1, -2.2, 3.3],
                              [4.4, 0.0, -5.5]], dtype=torch.float64).numpy()
    dim = 1
    keepdim = False
    out = make_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 3
    input_arr = torch.arange(16, dtype=torch.int32).reshape(4, 4).numpy()
    dim = 0
    keepdim = True
    out = make_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 4
    input_arr = (torch.arange(30, dtype=torch.float16).reshape(3, 2, 5) * 0.1).numpy()
    dim = -1
    keepdim = False
    out = make_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 5
    input_arr = (torch.arange(-24, 0, dtype=torch.int8).reshape(2, 3, 4) + 10).numpy()
    dim = 1
    keepdim = True
    out = make_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 6
    input_arr = torch.arange(2*3*4*5, dtype=torch.uint8).reshape(2, 3, 4, 5).numpy()
    dim = 2
    keepdim = False
    out = make_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 7
    input_arr = torch.linspace(-5, 4, steps=10, dtype=torch.float32).reshape(5, 1, 2).numpy()
    dim = 0
    keepdim = False
    out = make_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 8
    input_arr = torch.tensor([[-1], [0], [2]], dtype=torch.int64).numpy()
    dim = -2
    keepdim = True
    out = make_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 9
    input_arr = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3, 1).numpy()
    dim = -1
    keepdim = True
    out = make_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 10
    input_arr = torch.arange(2*1*3*2*2, dtype=torch.float32).reshape(2, 1, 3, 2, 2).numpy()
    dim = 2
    keepdim = False
    out = make_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 11
    input_arr = torch.tensor([-7, -7, -7, -8, 0], dtype=torch.int16).numpy()
    dim = 0
    keepdim = True
    out = make_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # 12
    input_arr = torch.tensor([[1.0, 2.0, 2.0],
                              [3.0, 3.0, 1.0],
                              [-1.0, -1.0, -1.0]], dtype=torch.float32).numpy()
    dim = 1
    keepdim = False
    out = make_out(input_arr, dim, keepdim)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    return list_of_inputs

generated_inputs["torch.max_2"] = torch_max_2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.max_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.max_2'.")


check_valid('torch.max', generated_inputs['torch.max_2'], lib="torch", suffix=2)
