
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def flipud_inputs():
    list_of_inputs = []

    # 1
    input = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 2
    input = torch.tensor([[-1, 0, 1], [2, -3, 4]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 3
    input = torch.arange(24, dtype=torch.float64).view(2, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 4
    input = torch.tensor([[True, False], [False, True], [True, True]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 5
    input = torch.tensor([1+2j, -3+0.5j, 0-1j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 6
    input = torch.arange(16, dtype=torch.int8).view(2, 2, 2, 2).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 7 non-contiguous slice
    base = torch.arange(24, dtype=torch.int32).view(4, 6).numpy()
    input = base[:, ::2]
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 8 transposed view
    input = torch.arange(12, dtype=torch.float32).view(3, 4).numpy().T
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 9 float16 with NaN/Inf
    input = torch.tensor([[[float('nan'), 1.0], [float('inf'), -float('inf')]]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 10 empty along first dimension
    input = torch.empty((0, 5), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 11 single column
    input = torch.arange(4, dtype=torch.int64).view(4, 1).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 12 5D tensor
    input = torch.randn((1, 2, 1, 3, 4), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 13 single-element 1D
    input = torch.tensor([42], dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 14 uint8 values
    input = torch.tensor([[255, 0], [128, 64]], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.flipud"] = flipud_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.flipud' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.flipud'.")


check_valid('torch.flipud', generated_inputs['torch.flipud'], lib="torch", suffix=0)
