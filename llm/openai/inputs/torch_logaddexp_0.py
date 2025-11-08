
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def logaddexp_inputs():
    list_of_inputs = []

    def add_case(input_arr, other_arr, out_arr=None):
        if out_arr is None:
            out_shape = np.broadcast(input_arr, other_arr).shape
            out_dtype = np.result_type(input_arr, other_arr)
            out_arr = np.empty(out_shape, dtype=out_dtype)
        list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    a1 = np.array([1.0, -2.0, 0.0], dtype=np.float32)
    b1 = np.array([0.5, 3.0, -1.5], dtype=np.float32)
    add_case(a1, b1)

    a2 = np.array([-100.0, -200.0, -300.0], dtype=np.float64)
    b2 = np.array(-1.0, dtype=np.float64)
    add_case(a2, b2)

    a3 = np.arange(-6, 0, dtype=np.float32).reshape(2, 3)
    b3 = np.array([[0.0, -1.0, 2.0]], dtype=np.float32)
    add_case(a3, b3)

    a4 = np.array([1e-3, 1e2, -1e2, 0.0], dtype=np.float16)
    b4 = np.array([2e-3, -1e2, 1e2, -0.0], dtype=np.float16)
    add_case(a4, b4)

    a5 = (np.arange(2 * 1 * 4, dtype=np.float32).reshape(2, 1, 4) * 0.1) - 1.0
    b5 = (np.arange(3, dtype=np.float32).reshape(1, 3, 1)) - 1.0
    add_case(a5, b5)

    a6 = np.array([[0.0, -np.inf], [1e308, -1e308]], dtype=np.float64)
    b6 = np.array([[np.inf, -np.inf], [-1e308, 1e308]], dtype=np.float64)
    add_case(a6, b6)

    a7 = np.empty((0, 3), dtype=np.float32)
    b7 = np.array([[0.0, 1.0, -1.0]], dtype=np.float32)
    add_case(a7, b7)

    a8 = np.array([1.0, 2000.0, 30000.0], dtype=np.float64)
    b8 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    add_case(a8, b8)

    a9 = np.array([[-1000.0], [-2000.0], [-3000.0]], dtype=np.float32)
    b9 = np.array([[0.0, -0.1, 0.1, 2.0]], dtype=np.float32)
    add_case(a9, b9)

    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    a10 = base[:, ::2] * 0.01
    b10 = (base[:, ::2] - 5.0) * -0.02
    add_case(a10, b10)

    a11 = np.array(0.0, dtype=np.float32)
    b11 = np.array(-np.inf, dtype=np.float32)
    add_case(a11, b11)

    a12 = (np.arange(1 * 2 * 1 * 3, dtype=np.float32).reshape(1, 2, 1, 3) * 0.1)
    b12 = (np.arange(2 * 1 * 4 * 1, dtype=np.float32).reshape(2, 1, 4, 1) * -0.05)
    add_case(a12, b12)

    a13 = np.array([[1.0, -2.0]], dtype=np.float16)
    b13 = np.array([[3.0, 4.0]], dtype=np.float32)
    add_case(a13, b13)

    return list_of_inputs

generated_inputs["torch.logaddexp"] = logaddexp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logaddexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logaddexp'.")


check_valid('torch.logaddexp', generated_inputs['torch.logaddexp'], lib="torch", suffix=0)
