
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def arccosh_inputs():
    list_of_inputs = []

    # 1: 0-D float32 scalar
    input = torch.tensor(1.0, dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 2: 1-D float64 vector
    input = torch.tensor([1.0, 1.0001, 2.0, 3.5], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 3: 2-D float16 matrix
    input = torch.tensor([[1.0, 2.0], [10.0, 100.0]], dtype=torch.float16).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 4: 3-D float32 tensor
    input = torch.tensor([[[1.0, 1.5, 2.0]], [[3.0, 4.0, 5.0]]], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 5: 1-D complex64 vector
    input = torch.tensor([1+0j, 0.5+0.5j, -2+3j, 4-1j], dtype=torch.complex64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 6: 2-D complex128 matrix
    input = torch.tensor([[1+0j, -1+2j], [3-4j, 0.1+0.2j]], dtype=torch.complex128).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 7: Empty 1-D float32
    input = torch.empty((0,), dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 8: Empty 3-D float64
    input = torch.empty((2, 0, 3), dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 9: Strided 1-D float32 (slice)
    base = torch.linspace(1, 10, steps=10, dtype=torch.float32).numpy()
    input = base[::2]
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 10: Includes inf and nan, float64
    input = torch.tensor([float('inf'), 1.0, float('nan'), 2.0], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 11: Fortran-ordered 2-D float32
    arr = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3).numpy()
    input = np.asfortranarray(arr)
    out = np.empty_like(input, order='F')
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 12: 4-D float32 with varied magnitudes
    vals = torch.tensor([1.0, 1.5, 100.0, 1e10, 2.0, 1.000001, 3.0, 5.0, 10.0, 1.0001, 7.5, 2.5], dtype=torch.float32).numpy()
    input = vals.reshape(2, 2, 1, 3)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 13: Real values including <1 to produce NaNs, float32
    input = torch.tensor([0.5, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.arccosh"] = arccosh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.arccosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arccosh'.")

check_valid('torch.arccosh', generated_inputs['torch.arccosh'], lib="torch", suffix=0)
