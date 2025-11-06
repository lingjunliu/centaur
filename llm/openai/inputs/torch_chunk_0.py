
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def chunk_inputs():
    list_of_inputs = []

    input = torch.arange(11, dtype=torch.int32).numpy()
    chunks = 6
    dim = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "chunks": chunks, "dim": dim}))

    input = torch.arange(12, dtype=torch.float32).numpy()
    chunks = 6
    dim = -1
    list_of_inputs.append(copy.deepcopy({"input": input, "chunks": chunks, "dim": dim}))

    input = torch.tensor([1+2j, 3+4j, 5+6j, 7+8j, 9+10j], dtype=torch.complex64).numpy()
    chunks = 2
    dim = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "chunks": chunks, "dim": dim}))

    input = torch.arange(24, dtype=torch.float64).reshape(4, 6).numpy()
    chunks = 3
    dim = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "chunks": chunks, "dim": dim}))

    input = torch.arange(24, dtype=torch.int64).reshape(4, 6).numpy()
    chunks = 4
    dim = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "chunks": chunks, "dim": dim}))

    input = (torch.randint(0, 2, (7, 3)) > 0).numpy()
    chunks = 2
    dim = -2
    list_of_inputs.append(copy.deepcopy({"input": input, "chunks": chunks, "dim": dim}))

    input = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    chunks = 2
    dim = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "chunks": chunks, "dim": dim}))

    input = torch.randint(-128, 127, (5, 1, 8), dtype=torch.int8).numpy()
    chunks = 4
    dim = -1
    list_of_inputs.append(copy.deepcopy({"input": input, "chunks": chunks, "dim": dim}))

    input = torch.randint(-32768, 32767, (6, 7, 8), dtype=torch.int16).numpy()
    chunks = 6
    dim = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "chunks": chunks, "dim": dim}))

    input = torch.randn(3, 4, 5, 6, dtype=torch.float32).numpy()
    chunks = 5
    dim = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "chunks": chunks, "dim": dim}))

    input = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    chunks = 2
    dim = -4
    list_of_inputs.append(copy.deepcopy({"input": input, "chunks": chunks, "dim": dim}))

    input = (torch.randn(2, 2, 2, 2, 9, dtype=torch.float64) + 1j * torch.randn(2, 2, 2, 2, 9, dtype=torch.float64)).to(torch.complex128).numpy()
    chunks = 4
    dim = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "chunks": chunks, "dim": dim}))

    input = torch.arange(10, dtype=torch.int16).numpy()
    chunks = 10
    dim = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "chunks": chunks, "dim": dim}))

    return list_of_inputs

generated_inputs["torch.chunk"] = chunk_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.chunk' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.chunk'.")


check_valid('torch.chunk', generated_inputs['torch.chunk'], lib="torch", suffix=0)
