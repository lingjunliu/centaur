
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def mul_inputs():
    list_of_inputs = []

    # Input 1: float32 1D
    input = torch.tensor([1.0, -2.5, 3.3], dtype=torch.float32).numpy()
    other = torch.tensor([4.0, 0.5, -1.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 2: broadcasting with 0-d float32
    input = torch.arange(6, dtype=torch.float32).reshape(2, 3).numpy()
    other = torch.tensor(2.0, dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 3: int32 2x2
    input = torch.tensor([[1, -2], [3, -4]], dtype=torch.int32).numpy()
    other = torch.tensor([[5, 6], [-7, 8]], dtype=torch.int32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 4: mixed int32 and float64 with broadcasting
    input = torch.tensor([[1], [2]], dtype=torch.int32).numpy()
    other = torch.tensor([[0.5, -1.5, 2.0]], dtype=torch.float64).numpy()
    out = np.empty((2, 3), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 5: complex64 and float32
    input = torch.tensor([1+2j, -3+0.5j, -1j], dtype=torch.complex64).numpy()
    other = torch.tensor([2.0, -0.5, 3.0], dtype=torch.float32).numpy()
    out = np.empty((3,), dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 6: complex128 and complex64 with broadcasting
    input = torch.tensor([[[1+1j, -2-2j, 3j]]], dtype=torch.complex128).numpy()
    other = torch.tensor([1-1j, 2+0j, -1+2j], dtype=torch.complex64).numpy()
    out = np.empty((1, 1, 3), dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 7: zero-size dimension
    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    other = torch.empty((0, 1), dtype=torch.float32).numpy()
    out = np.empty((0, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 8: high-dimensional broadcasting
    input = torch.randn(2, 3, 1, 4, dtype=torch.float32).numpy()
    other = torch.randn(1, 3, 5, 1, dtype=torch.float32).numpy()
    out = np.empty((2, 3, 5, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 9: uint8 and int16
    input = torch.tensor([[1, 200, 255]], dtype=torch.uint8).numpy()
    other = torch.tensor([[2, -1, 3]], dtype=torch.int16).numpy()
    out = np.empty((1, 3), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 10: float64 with NaN and Inf
    input = torch.tensor([np.nan, np.inf, -np.inf, 1.5], dtype=torch.float64).numpy()
    other = torch.tensor([1.0, -1.0, 0.0, 2.0], dtype=torch.float64).numpy()
    out = np.empty((4,), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 11: int64 with 0-d scalar broadcasting
    input = torch.tensor([10, -20, 0, 7], dtype=torch.int64).numpy()
    other = torch.tensor(-3, dtype=torch.int64).numpy()
    out = np.empty((4,), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 12: complex64 with 0-d complex scalar
    input = torch.tensor([[1+0j, 0+1j], [-1-1j, 2+2j]], dtype=torch.complex64).numpy()
    other = torch.tensor(0.5+0.5j, dtype=torch.complex64).numpy()
    out = np.empty((2, 2), dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 13: float16 broadcasting
    input = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float16).numpy()
    other = torch.tensor([[0.5], [2.0]], dtype=torch.float16).numpy()
    out = np.empty((2, 3), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 14: int16 broadcasting with extremes
    input = torch.tensor([[-32768], [32767]], dtype=torch.int16).numpy()
    other = torch.tensor([[-1, 2]], dtype=torch.int16).numpy()
    out = np.empty((2, 2), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.mul_1"] = mul_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.mul_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mul_1'.")


check_valid('torch.mul', generated_inputs['torch.mul_1'], lib="torch", suffix=1)
