
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def asin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 within domain
    input_arr = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32).numpy()
    out_arr = torch.empty_like(torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 2: 2D float64 within domain
    input_arr = torch.tensor([[-1.0, -0.25, 0.0],
                              [0.25, 0.75, 1.0]], dtype=torch.float64).numpy()
    out_arr = torch.empty((2, 3), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 3: values outside domain (float32)
    input_arr = torch.tensor([1.2, -1.5, 0.0], dtype=torch.float32).numpy()
    out_arr = torch.empty(3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 4: scalar 0-D float32
    input_arr = torch.tensor(0.3, dtype=torch.float32).numpy()
    out_arr = torch.empty((), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 5: empty array
    input_arr = torch.empty(0, dtype=torch.float32).numpy()
    out_arr = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 6: integer input (int64), result expected float32
    input_arr = torch.tensor([-1, 0, 1, 2], dtype=torch.int64).numpy()
    out_arr = torch.empty(4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 7: complex64 2x2
    input_arr = torch.tensor([[0+0j, 1+0j],
                              [0+1j, -0.5+0.5j]], dtype=torch.complex64).numpy()
    out_arr = torch.empty((2, 2), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 8: complex128 3D
    input_arr = torch.tensor([[[0.2+0.1j, -0.3+0.4j, 0-1j]] ,
                              [[1+0j, -1+0j, 0.5-0.5j]]], dtype=torch.complex128).numpy()
    out_arr = torch.empty((2, 1, 3), dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 9: 4D float32 random in [-1, 1]
    input_arr = ((torch.rand(2, 2, 2, 2, dtype=torch.float32) * 2) - 1).numpy()
    out_arr = torch.empty((2, 2, 2, 2), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 10: non-contiguous via transpose (float64)
    base = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    input_arr = base.t().numpy()
    out_arr = torch.empty((3, 2), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 11: integer input (int32) 2x2, result expected float32
    input_arr = torch.tensor([[-1, 0],
                              [1, 2]], dtype=torch.int32).numpy()
    out_arr = torch.empty((2, 2), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 12: float64 with NaN and Inf
    input_arr = torch.tensor([float('nan'), float('inf'), float('-inf')], dtype=torch.float64).numpy()
    out_arr = torch.empty(3, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 13: values near edges of domain (float32)
    input_arr = torch.tensor([-1.0 + 1e-7, 1.0 - 1e-7, 0.9999999, -0.9999999], dtype=torch.float32).numpy()
    out_arr = torch.empty(4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.asin"] = asin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.asin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.asin'.")


check_valid('torch.asin', generated_inputs['torch.asin'], lib="torch", suffix=0)
