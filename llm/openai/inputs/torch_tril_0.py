
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def tril_inputs():
    list_of_inputs = []

    # Input 1: 3x3 float32, diagonal=0
    inp = np.arange(9, dtype=np.float32).reshape(3, 3)
    out = np.empty_like(inp)
    input_dict = {"input": inp, "diagonal": 0, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x5 float64, diagonal=1
    inp = (np.arange(15, dtype=np.float64).reshape(3, 5) - 7.5)
    out = np.zeros_like(inp)
    input_dict = {"input": inp, "diagonal": np.int32(1), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4x4 int64, diagonal=-1
    inp = np.arange(16, dtype=np.int64).reshape(4, 4)
    out = np.empty_like(inp)
    input_dict = {"input": inp, "diagonal": -1, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2x3 int32, diagonal=-2
    inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    out = np.zeros_like(inp)
    input_dict = {"input": inp, "diagonal": np.int64(-2), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3x3 complex64, diagonal=2
    base = np.arange(9, dtype=np.float32).reshape(3, 3)
    inp = base + 1j * (base + 0.5).astype(np.complex64)
    inp = inp.astype(np.complex64)
    out = np.empty_like(inp)
    input_dict = {"input": inp, "diagonal": 2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: batch 2x3 of 4x4 float32, diagonal=0
    inp = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4) / 10.0
    out = np.zeros_like(inp)
    input_dict = {"input": inp, "diagonal": 0, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 5x3 bool, diagonal=0
    inp = (np.arange(15) % 2 == 0).reshape(5, 3)
    out = np.zeros_like(inp)
    input_dict = {"input": inp, "diagonal": np.int32(0), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1x1 float16, diagonal=-1
    inp = np.array([[3.5]], dtype=np.float16)
    out = np.empty_like(inp)
    input_dict = {"input": inp, "diagonal": -1, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 10x10 float64, diagonal=100 (large positive)
    inp = np.linspace(-1.0, 1.0, 100, dtype=np.float64).reshape(10, 10)
    out = np.empty_like(inp)
    input_dict = {"input": inp, "diagonal": 100, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 6x7 uint8, diagonal=-100 (large negative)
    inp = (np.arange(42, dtype=np.uint8).reshape(6, 7))
    out = np.zeros_like(inp)
    input_dict = {"input": inp, "diagonal": -100, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3x3 complex128, diagonal=1
    base = np.arange(9, dtype=np.float64).reshape(3, 3)
    inp = (base - 4) + 1j * (base + 2)
    inp = inp.astype(np.complex128)
    out = np.empty_like(inp)
    input_dict = {"input": inp, "diagonal": np.int64(1), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: batch 4x5 matrices in 3D tensor: shape (4, 5, 3, 6) float32, diagonal=-2
    inp = np.arange(4 * 5 * 3 * 6, dtype=np.float32).reshape(4, 5, 3, 6) / 5.0
    out = np.empty_like(inp)
    input_dict = {"input": inp, "diagonal": -2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.tril"] = tril_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.tril' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tril'.")


check_valid('torch.tril', generated_inputs['torch.tril'], lib="torch", suffix=0)
