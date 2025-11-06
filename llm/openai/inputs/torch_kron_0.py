
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def kron_inputs():
    def compute_out(a, b):
        a_shape, b_shape = a.shape, b.shape
        la, lb = len(a_shape), len(b_shape)
        if la < lb:
            a_shape = (1,) * (lb - la) + a_shape
        elif lb < la:
            b_shape = (1,) * (la - lb) + b_shape
        out_shape = tuple(int(ai) * int(bi) for ai, bi in zip(a_shape, b_shape))
        out_dtype = np.result_type(a, b)
        return np.empty(out_shape, dtype=out_dtype)

    list_of_inputs = []

    a = torch.eye(2, dtype=torch.float32).numpy()
    b = np.ones((2, 2), dtype=np.float32)
    out = compute_out(a, b)
    list_of_inputs.append(copy.deepcopy({"input": a, "other": b, "out": out}))

    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([4, -5], dtype=np.int64)
    out = compute_out(a, b)
    list_of_inputs.append(copy.deepcopy({"input": a, "other": b, "out": out}))

    a = np.arange(2 * 1 * 3, dtype=np.float64).reshape(2, 1, 3)
    b = (np.arange(1 * 3 * 2, dtype=np.int32).reshape(1, 3, 2) - 3)
    out = compute_out(a, b)
    list_of_inputs.append(copy.deepcopy({"input": a, "other": b, "out": out}))

    a = np.array([[2, -1, 0], [3, 5, -2]], dtype=np.int16)
    b = np.array([[1.5, -0.5]], dtype=np.float32)
    out = compute_out(a, b)
    list_of_inputs.append(copy.deepcopy({"input": a, "other": b, "out": out}))

    a = np.array([-1.0, 0.0, 2.5, -3.5], dtype=np.float64)
    b = np.array([[1, 2], [3, 4]], dtype=np.int64)
    out = compute_out(a, b)
    list_of_inputs.append(copy.deepcopy({"input": a, "other": b, "out": out}))

    a = np.arange(9, dtype=np.float16).reshape(3, 3)
    b = np.array([1, -1, 2, 0, -3], dtype=np.int8)
    out = compute_out(a, b)
    list_of_inputs.append(copy.deepcopy({"input": a, "other": b, "out": out}))

    a = np.array([[1 + 2j], [3 - 1j]], dtype=np.complex64)
    b = np.array([[0 + 1j, -2 + 0j, 3 + 4j]], dtype=np.complex64)
    out = compute_out(a, b)
    list_of_inputs.append(copy.deepcopy({"input": a, "other": b, "out": out}))

    a = np.array(3.0, dtype=np.float32)
    b = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = compute_out(a, b)
    list_of_inputs.append(copy.deepcopy({"input": a, "other": b, "out": out}))

    a = (np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) - 5)
    b = np.linspace(1.0, 2.0, 5, dtype=np.float32)
    out = compute_out(a, b)
    list_of_inputs.append(copy.deepcopy({"input": a, "other": b, "out": out}))

    a = (np.arange(2 * 2 * 1 * 3, dtype=np.float64).reshape(2, 2, 1, 3) + 0.5)
    b = (np.arange(1 * 3 * 2 * 2, dtype=np.float32).reshape(1, 3, 2, 2) - 1.0)
    out = compute_out(a, b)
    list_of_inputs.append(copy.deepcopy({"input": a, "other": b, "out": out}))

    a = np.arange(1 * 2 * 3 * 4 * 1, dtype=np.int32).reshape(1, 2, 3, 4, 1)
    b = (np.arange(2 * 1 * 1 * 1 * 5, dtype=np.int32).reshape(2, 1, 1, 1, 5) + 1)
    out = compute_out(a, b)
    list_of_inputs.append(copy.deepcopy({"input": a, "other": b, "out": out}))

    real = np.linspace(0, 7, 8, dtype=np.float32).reshape(2, 2, 2)
    imag = np.linspace(1, 8, 8, dtype=np.float32).reshape(2, 2, 2)
    b = (real + 1j * imag).astype(np.complex64)
    a = np.array([[-1.0, 2.0, 0.5],
                  [3.0, -4.0, 1.5],
                  [2.5, 0.0, -2.0]], dtype=np.float32)
    out = compute_out(a, b)
    list_of_inputs.append(copy.deepcopy({"input": a, "other": b, "out": out}))

    return list_of_inputs

generated_inputs["torch.kron"] = kron_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.kron' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.kron'.")


check_valid('torch.kron', generated_inputs['torch.kron'], lib="torch", suffix=0)
