
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clamp_inputs():
    list_of_inputs = []

    # 1
    input_t = torch.tensor([-1.2, 0.3, 2.0, -0.7], dtype=torch.float32)
    min_t = torch.tensor(0.0, dtype=torch.float32)
    out_t = torch.empty_like(input_t)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "min": min_t.numpy(), "max": None, "out": out_t.numpy()}))

    # 2
    input_t = torch.tensor([[1.5, -2.0, 0.3],
                            [4.2, 5.5, -0.1]], dtype=torch.float64)
    min_t = torch.tensor([[0.0, -1.0, 0.2],
                          [1.0, 2.0, -0.5]], dtype=torch.float64)
    out_t = torch.empty_like(input_t)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "min": min_t.numpy(), "max": None, "out": out_t.numpy()}))

    # 3
    input_t = torch.tensor([[[-10, -3, 0]],
                            [[4, 6, 9]]], dtype=torch.int32)
    min_t = torch.tensor(-5, dtype=torch.int32)
    out_t = torch.empty_like(input_t)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "min": min_t.numpy(), "max": None, "out": out_t.numpy()}))

    # 4
    input_t = torch.randn(2, 3, 4, dtype=torch.float32)
    min_t = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32).view(1, 3, 1)
    out_t = torch.empty_like(input_t)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "min": min_t.numpy(), "max": None, "out": out_t.numpy()}))

    # 5
    input_t = torch.tensor(-0.3, dtype=torch.float16)
    min_t = torch.tensor(-1.0, dtype=torch.float16)
    out_t = torch.empty_like(input_t)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "min": min_t.numpy(), "max": None, "out": out_t.numpy()}))

    # 6
    input_t = torch.tensor([-3.0, -1.0, 0.0, 4.0, 9.0], dtype=torch.float32)
    min_t = torch.full((5,), 5.0, dtype=torch.float32)
    out_t = torch.empty_like(input_t)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "min": min_t.numpy(), "max": None, "out": out_t.numpy()}))

    # 7
    input_t = torch.tensor([0, 128, 255], dtype=torch.uint8)
    min_t = torch.tensor([10, 0, 250], dtype=torch.uint8)
    out_t = torch.empty_like(input_t)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "min": min_t.numpy(), "max": None, "out": out_t.numpy()}))

    # 8
    input_t = torch.tensor([[-10, -1],
                            [0, 5]], dtype=torch.int64)
    min_t = torch.tensor([[-100, -2],
                          [-1, 0]], dtype=torch.int64)
    out_t = torch.empty_like(input_t)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "min": min_t.numpy(), "max": None, "out": out_t.numpy()}))

    # 9
    input_t = torch.randn(1, 2, 3, 4, dtype=torch.float32)
    min_t = torch.tensor(0.0, dtype=torch.float32)
    out_t = torch.empty_like(input_t)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "min": min_t.numpy(), "max": None, "out": out_t.numpy()}))

    # 10
    input_t = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.1, 0.1], dtype=torch.float32)
    min_t = torch.tensor(-1.0, dtype=torch.float32)
    out_t = torch.empty_like(input_t)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "min": min_t.numpy(), "max": None, "out": out_t.numpy()}))

    # 11
    base = torch.arange(24, dtype=torch.float32).view(4, 6)
    input_t = base[:, ::2]
    min_t = torch.tensor([[-1.0], [0.0], [1.0], [2.0]], dtype=torch.float32)
    out_t = torch.empty_like(input_t)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "min": min_t.numpy(), "max": None, "out": out_t.numpy()}))

    # 12
    input_t = torch.empty(0, 5, dtype=torch.float32)
    min_t = torch.tensor(-1.0, dtype=torch.float32)
    out_t = torch.empty_like(input_t)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "min": min_t.numpy(), "max": None, "out": out_t.numpy()}))

    return list_of_inputs

generated_inputs["torch.clamp_2"] = clamp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.clamp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clamp_2'.")


check_valid('torch.clamp', generated_inputs['torch.clamp_2'], lib="torch", suffix=2)
