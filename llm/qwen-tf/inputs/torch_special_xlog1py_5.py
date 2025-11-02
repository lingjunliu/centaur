
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def xlog1py_inputs():
    list_of_inputs = []

    input_val = np.int32(3)
    other = np.array([0.0, 1.0, -0.5], dtype=np.float32)
    out = np.empty_like(other, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(-2)
    other = np.array([[1.0, 2.0], [3.0, -0.9]], dtype=np.float64)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int16(0)
    other = np.array(2.5, dtype=np.float64)
    out = np.empty((), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int8(1)
    other = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = np.zeros_like(other, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    rng = np.random.RandomState(0)
    input_val = np.int64(7)
    other = rng.randn(3, 1, 4).astype(np.float64)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int32(-5)
    other = np.array([-1.5, -1.0, -0.999, 0.0, 1.0], dtype=np.float64)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(2)
    other = np.array([[True, False], [False, True]], dtype=bool)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(4)
    other = np.array([10, 0, -1, 3], dtype=np.int32)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(1)
    other = np.linspace(-0.9999999, 100.0, 8, dtype=np.float64)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(-1)
    other = np.zeros((0,), dtype=np.float32)
    out = np.empty_like(other, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(123456789)
    other = np.full((2, 2, 2), 1e-12, dtype=np.float64)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(3)
    other = np.array([np.inf, -np.inf, np.nan], dtype=np.float64)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.xlog1py_5"] = xlog1py_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.xlog1py_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.xlog1py_5'.")


check_valid('torch.special.xlog1py', generated_inputs['torch.special.xlog1py_5'], lib="torch", suffix=5)
