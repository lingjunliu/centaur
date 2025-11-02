
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def xlog1py_inputs():
    list_of_inputs = []

    # Input 1
    input_val = 0.0
    other = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 2
    input_val = 2.5
    other = np.array([[-0.5, 0.0], [1.5, 10.0]], dtype=np.float64)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 3
    input_val = -3.0
    other = np.linspace(-0.9, 0.9, 7, dtype=np.float32)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 4
    input_val = np.float64(1.0)
    other = torch.linspace(-0.99, 5.0, steps=12, dtype=torch.float64).reshape(3, 4).numpy()
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 5
    input_val = 1e-6
    other = np.array([0.0, -1e-4, 1e-4, 10.0, 1000.0], dtype=np.float16)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 6
    input_val = 2.0
    other = np.array([-1.0, -0.999, -0.5, 0.5], dtype=np.float64)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 7
    input_val = 12345.6789
    other = np.full((3, 3), 1e-8, dtype=np.float32)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 8
    input_val = 1.5
    other = np.array([1e10, 1e20, 1e-10], dtype=np.float64)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 9
    input_val = np.float32(-2.5)
    other = torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32).numpy()
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 10
    input_val = 3.14159
    other = np.array(0.5, dtype=np.float64)
    out = np.empty((), dtype=other.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 11
    input_val = -7.0
    base = np.arange(12, dtype=np.float32).reshape(4, 3)
    other = base[:, ::-1]
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 12
    input_val = 0.75
    rng = np.random.RandomState(0)
    other = rng.uniform(-0.9, 2.0, size=(1, 2, 1, 3)).astype(np.float32)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.xlog1py_4"] = xlog1py_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.xlog1py_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.xlog1py_4'.")


check_valid('torch.special.xlog1py', generated_inputs['torch.special.xlog1py_4'], lib="torch", suffix=4)
