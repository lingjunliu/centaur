
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def amin_inputs():
    def reduced_shape(shape, dim, keepdim):
        d = int(dim)
        if d < 0:
            d += len(shape)
        if keepdim:
            return tuple(1 if i == d else s for i, s in enumerate(shape))
        else:
            return tuple(s for i, s in enumerate(shape) if i != d)

    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(3, 4, dtype=torch.float32).numpy()
    dim = np.int64(1)
    keepdim = False
    out = np.empty(reduced_shape(input_arr.shape, dim, keepdim), dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 2
    input_arr = torch.randn(4, 1, dtype=torch.float64).numpy()
    dim = np.int32(0)
    keepdim = True
    out = np.empty(reduced_shape(input_arr.shape, dim, keepdim), dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 3
    input_arr = torch.randint(-10, 10, (2, 3, 4), dtype=torch.int64).numpy()
    dim = np.int64(2)
    keepdim = False
    out = np.empty(reduced_shape(input_arr.shape, dim, keepdim), dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 4
    input_arr = torch.tensor([5, -3, 7, -1, 2], dtype=torch.int32).numpy()
    dim = np.int32(0)
    keepdim = True
    out = np.empty(reduced_shape(input_arr.shape, dim, keepdim), dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 5
    input_arr = (torch.rand(2, 2, 3) > 0.5).numpy()
    dim = np.int64(-1)
    keepdim = False
    out = np.empty(reduced_shape(input_arr.shape, dim, keepdim), dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 6
    input_arr = torch.randn(3, 1, 5, dtype=torch.float16).numpy()
    dim = np.int32(0)
    keepdim = True
    out = np.empty(reduced_shape(input_arr.shape, dim, keepdim), dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 7
    input_arr = torch.randint(0, 255, (4, 2), dtype=torch.uint8).numpy()
    dim = np.int64(-2)
    keepdim = False
    out = np.empty(reduced_shape(input_arr.shape, dim, keepdim), dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 8
    input_arr = torch.randn(2, 3, 2, dtype=torch.float32).numpy()
    input_arr[0, 0, 0] = np.nan
    dim = np.int32(1)
    keepdim = True
    out = np.empty(reduced_shape(input_arr.shape, dim, keepdim), dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 9
    input_arr = torch.randn(2, 2, 2, 2, 2, dtype=torch.float32).numpy()
    dim = np.int64(3)
    keepdim = True
    out = np.empty(reduced_shape(input_arr.shape, dim, keepdim), dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 10
    input_arr = torch.tensor([[-10, -5, -3, -20, -1],
                              [7, -8, 2, 0, -15]], dtype=torch.int16).numpy()
    dim = np.int32(0)
    keepdim = False
    out = np.empty(reduced_shape(input_arr.shape, dim, keepdim), dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 11
    input_arr = (torch.rand(2, 3, 4, 5) > 0.3).numpy()
    dim = np.int64(2)
    keepdim = True
    out = np.empty(reduced_shape(input_arr.shape, dim, keepdim), dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 12
    input_arr = torch.linspace(-5.0, 5.0, steps=8, dtype=torch.float64).numpy()
    dim = np.int32(-1)
    keepdim = False
    out = np.empty(reduced_shape(input_arr.shape, dim, keepdim), dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    return list_of_inputs

generated_inputs["torch.amin_1"] = amin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.amin_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.amin_1'.")


check_valid('torch.amin', generated_inputs['torch.amin_1'], lib="torch", suffix=1)
