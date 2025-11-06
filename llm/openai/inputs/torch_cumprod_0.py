
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cumprod_inputs():
    list_of_inputs = []

    inp = torch.tensor([1.0, 2.0, 3.0, 4.0, -5.0], dtype=torch.float32).numpy()
    dim = 0
    dtype = np.dtype('float32')
    out = np.empty(inp.shape, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "dtype": dtype, "out": out}))

    inp = torch.tensor([3, -2, 4, -1, 0, 5], dtype=torch.int32).numpy()
    dim = 0
    dtype = np.dtype('int64')
    out = np.empty(inp.shape, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "dtype": dtype, "out": out}))

    inp = torch.tensor([[1.5, -2.0, 3.0],
                        [0.5, 2.0, -1.0]], dtype=torch.float64).numpy()
    dim = 1
    dtype = np.dtype('float64')
    out = np.zeros(inp.shape, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "dtype": dtype, "out": out}))

    inp = torch.tensor([[1, -2],
                        [3, 4],
                        [-5, 6]], dtype=torch.int8).numpy()
    dim = 0
    dtype = np.dtype('int16')
    out = np.empty(inp.shape, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "dtype": dtype, "out": out}))

    inp = torch.randn(2, 2, 3, dtype=torch.float16).numpy()
    dim = -1
    dtype = np.dtype('float16')
    out = np.empty(inp.shape, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "dtype": dtype, "out": out}))

    inp = torch.tensor([[[1, 2, 0, -1],
                         [2, -1, 3, 1],
                         [1, 1, -2, 2]],
                        [[-1, 2, 2, 2],
                         [3, 0, -1, 1],
                         [2, 1, 1, -1]]], dtype=torch.int64).numpy()
    dim = 1
    dtype = np.dtype('int64')
    out = np.zeros(inp.shape, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "dtype": dtype, "out": out}))

    inp = torch.randn(2, 1, 3, 4, dtype=torch.float32).numpy()
    dim = -2
    dtype = np.dtype('float32')
    out = np.empty(inp.shape, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "dtype": dtype, "out": out}))

    real = torch.tensor([[1.0, -2.0, 0.5],
                         [0.0, 3.0, -1.5],
                         [2.0, -1.0, 1.0]], dtype=torch.float32)
    imag = torch.tensor([[0.5, 1.0, -0.5],
                         [2.0, 0.0, 0.5],
                         [-1.0, 1.5, 0.0]], dtype=torch.float32)
    inp = torch.complex(real, imag).numpy()
    dim = 0
    dtype = np.dtype('complex64')
    out = np.empty(inp.shape, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "dtype": dtype, "out": out}))

    base = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    inp = base.t().numpy()
    dim = 1
    dtype = np.dtype('float32')
    out = np.empty(inp.shape, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "dtype": dtype, "out": out}))

    inp = torch.tensor([[[1.0, -1.0],
                         [0.0, 2.0]],
                        [[-2.0, 3.0],
                         [4.0, 0.0]]], dtype=torch.float32).numpy()
    dim = 0
    dtype = np.dtype('float64')
    out = np.zeros(inp.shape, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "dtype": dtype, "out": out}))

    inp = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.uint8).numpy()
    dim = 0
    dtype = np.dtype('uint8')
    out = np.empty(inp.shape, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "dtype": dtype, "out": out}))

    inp = torch.randn(2, 1, 2, 1, 3, dtype=torch.float64).numpy()
    dim = 4
    dtype = np.dtype('float64')
    out = np.empty(inp.shape, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim, "dtype": dtype, "out": out}))

    return list_of_inputs

generated_inputs["torch.cumprod"] = cumprod_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cumprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cumprod'.")


check_valid('torch.cumprod', generated_inputs['torch.cumprod'], lib="torch", suffix=0)
