
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def ldexp_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 with mixed exponents
    input_arr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    other_arr = torch.tensor([1, -1, 0], dtype=torch.int32).numpy()
    out_arr = torch.empty(3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 2: Scalars (0-D arrays)
    input_arr = torch.tensor(1.5, dtype=torch.float64).numpy()
    other_arr = torch.tensor(10, dtype=torch.int64).numpy()
    out_arr = torch.empty((), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 3: Broadcasting (2,3) with (3,)
    input_arr = torch.tensor([[1.0, 0.5, 2.0],
                              [3.0, 4.0, 5.0]], dtype=torch.float32).numpy()
    other_arr = torch.tensor([1, 2, -3], dtype=torch.int32).numpy()
    out_arr = torch.empty((2, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 4: Broadcasting (3,1) with (1,4)
    input_arr = torch.tensor([[1.0],
                              [0.25],
                              [2.0]], dtype=torch.float64).numpy()
    other_arr = torch.tensor([[4, -2, 3, 0]], dtype=torch.int8).numpy()
    out_arr = torch.empty((3, 4), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 5: float16 with int16 exponents
    input_arr = torch.tensor([-0.5, 0.25, -1.0, 2.0], dtype=torch.float16).numpy()
    other_arr = torch.tensor([1, 2, -2, 0], dtype=torch.int16).numpy()
    out_arr = torch.empty(4, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 6: 3D broadcasting (2,1,3) with (1,4,1)
    input_arr = torch.tensor([[[1.0, 2.0, 3.0]],
                              [[4.0, 5.0, 6.0]]], dtype=torch.float32).numpy()
    other_arr = torch.tensor([[[0], [1], [-1], [2]]], dtype=torch.int32).numpy()
    out_arr = torch.empty((2, 4, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 7: complex64 with matching shape int32 exponents
    input_arr = torch.tensor([[1+1j, -2-0.5j],
                              [0.5+3j, -1+2j]], dtype=torch.complex64).numpy()
    other_arr = torch.tensor([[1, -2],
                              [3, 0]], dtype=torch.int32).numpy()
    out_arr = torch.empty((2, 2), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 8: complex128 with scalar exponent
    input_arr = torch.tensor([1+0j, -1-1j, 0.5+2j, -3+0.25j, 2-2j], dtype=torch.complex128).numpy()
    other_arr = torch.tensor(2, dtype=torch.int64).numpy()
    out_arr = torch.empty(5, dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 9: Very large exponent (may overflow)
    input_arr = torch.tensor([1.0], dtype=torch.float32).numpy()
    other_arr = torch.tensor([100], dtype=torch.int32).numpy()
    out_arr = torch.empty(1, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 10: Empty arrays
    input_arr = torch.empty(0, dtype=torch.float32).numpy()
    other_arr = torch.empty(0, dtype=torch.int32).numpy()
    out_arr = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 11: Full broadcasting from (1,1,1) and (3,4,5)
    input_arr = torch.tensor([[[2.0]]], dtype=torch.float64).numpy()
    other_arr = torch.arange(3*4*5, dtype=torch.int32).reshape(3, 4, 5).numpy()
    out_arr = torch.empty((3, 4, 5), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 12: 4D tensor with scalar exponent
    input_arr = torch.arange(2*3*4*5, dtype=torch.float64).reshape(2, 3, 4, 5).numpy()
    other_arr = torch.tensor(3, dtype=torch.int32).numpy()
    out_arr = torch.empty((2, 3, 4, 5), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.ldexp"] = ldexp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.ldexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ldexp'.")


check_valid('torch.ldexp', generated_inputs['torch.ldexp'], lib="torch", suffix=0)
