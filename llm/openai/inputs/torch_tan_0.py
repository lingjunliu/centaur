
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np
import math

def tan_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32).numpy()
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    input_arr = torch.randn(2, 3, dtype=torch.float64).numpy()
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    input_arr = torch.tensor(0.3, dtype=torch.float32).numpy()
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    input_arr = torch.randn(2, 2, 2, dtype=torch.float16).numpy()
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    real = torch.tensor([[1.0, 2.5], [-3.0, 0.0]], dtype=torch.float32)
    imag = torch.tensor([[0.5, -0.5], [1.0, -1.0]], dtype=torch.float32)
    input_arr = torch.complex(real, imag).numpy()
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    real = torch.tensor([math.pi, -math.pi / 2, 0.0], dtype=torch.float64)
    imag = torch.tensor([1e-3, -2e-3, 3e-3], dtype=torch.float64)
    input_arr = torch.complex(real, imag).numpy()
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    input_arr = torch.empty((0,), dtype=torch.float32).numpy()
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    input_arr = torch.empty((2, 0, 3), dtype=torch.float32).numpy()
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    base = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    input_arr = base[..., ::2]
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    input_arr = np.array([math.pi / 2 - 1e-6, -math.pi / 2 + 1e-6, 3 * math.pi / 2 - 1e-6, -3 * math.pi / 2 + 1e-6], dtype=np.float64)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    input_arr = np.array([0.0, math.inf, -math.inf, math.nan], dtype=np.float64)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    input_arr = np.array([1000.0, -1000.0, 1e-6, -1e-6], dtype=np.float32)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    input_arr = np.array([1e-3, -1e-3, 5e-4, -5e-4], dtype=np.float16)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    input_arr = torch.linspace(-2.0, 2.0, steps=24, dtype=torch.float64).reshape(1, 2, 3, 4).numpy()
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    input_arr = torch.linspace(-3.0, 3.0, steps=15, dtype=torch.float32).reshape(3, 1, 5).numpy()
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.tan"] = tan_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.tan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tan'.")


check_valid('torch.tan', generated_inputs['torch.tan'], lib="torch", suffix=0)
