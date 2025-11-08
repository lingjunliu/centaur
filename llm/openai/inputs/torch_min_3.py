
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def min_inputs():
    list_of_inputs = []

    # 1: 1D float32, same shape
    input = torch.tensor([1.5, -2.0, 3.0], dtype=torch.float32).numpy()
    other = torch.tensor([0.5, -3.0, 4.5], dtype=torch.float32).numpy()
    out = torch.empty(3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 2: 2D int32, same shape
    input = torch.tensor([[1, 2, 3],
                          [4, 5, 6]], dtype=torch.int32).numpy()
    other = torch.tensor([[6, 5, 4],
                          [3, 2, 1]], dtype=torch.int32).numpy()
    out = torch.empty((2, 3), dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 3: 2D float64 with scalar (0-d tensor) broadcast
    input = torch.tensor([[1.0, -1.0, 2.0, -2.0],
                          [3.5, 4.5, -5.0, 6.0],
                          [7.0, -8.0, 9.0, -10.0]], dtype=torch.float64).numpy()
    other = torch.tensor(0.0, dtype=torch.float64).numpy()
    out = torch.empty((3, 4), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 4: 2D float32 with last-dim vector broadcast
    input = torch.tensor([[0.1, -0.2, 0.3],
                          [1.0, -1.5, 2.5]], dtype=torch.float32).numpy()
    other = torch.tensor([0.0, -0.1, 0.2], dtype=torch.float32).numpy()
    out = torch.empty((2, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 5: 3D float32 broadcasting (2,1,3) with (1,4,1) -> (2,4,3)
    input = torch.tensor([[[1.0, -2.0, 3.0]],
                          [[-1.0, 2.0, -3.0]]], dtype=torch.float32).numpy()  # (2,1,3)
    other = torch.tensor([[[0.5],
                           [-0.5],
                           [1.5],
                           [-1.5]]], dtype=torch.float32).numpy()  # (1,4,1)
    out = torch.empty((2, 4, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 6: 1D uint8 with scalar broadcast, out as empty placeholder
    input = torch.tensor([10, 200, 30, 255, 0], dtype=torch.uint8).numpy()
    other = torch.tensor([128], dtype=torch.uint8).numpy()
    out = torch.tensor([], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 7: 3D int16, same shape with negatives
    input = torch.tensor([[[-1, 2],
                           [3, -4]],
                          [[5, -6],
                           [-7, 8]]], dtype=torch.int16).numpy()
    other = torch.tensor([[[0, 1],
                           [2, -5]],
                          [[4, -7],
                           [-6, 9]]], dtype=torch.int16).numpy()
    out = torch.empty((2, 2, 2), dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 8: 2D float16 with NaN/Inf, out as empty placeholder
    input = torch.tensor([[float('nan'), 1.0, float('inf')],
                          [-float('inf'), -1.0, 0.0]], dtype=torch.float16).numpy()
    other = torch.tensor([[0.5, float('nan'), 2.0],
                          [1.0, -2.0, float('inf')]], dtype=torch.float16).numpy()
    out = torch.tensor([], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 9: 4D float32 broadcasting (1,3,4,5) with (2,1,1,5) -> (2,3,4,5)
    input = torch.arange(1*3*4*5, dtype=torch.float32).reshape(1, 3, 4, 5).numpy()
    other = torch.linspace(-1.0, 1.0, steps=5, dtype=torch.float32).reshape(1, 1, 1, 5).repeat(2, 1, 1, 1).numpy()
    out = torch.empty((2, 3, 4, 5), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 10: Empty dimension broadcasting (0,3) with (1,3)
    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    other = torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float32).numpy()
    out = torch.empty((0, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 11: int64 broadcasting (3,1) with (3,4) -> (3,4)
    input = torch.tensor([[10],
                          [-5],
                          [3]], dtype=torch.int64).numpy()
    other = torch.tensor([[7, 8, 9, 10],
                          [-10, -5, 0, 5],
                          [3, 2, 1, 0]], dtype=torch.int64).numpy()
    out = torch.empty((3, 4), dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 12: 1D int8, same shape with negatives
    input = torch.tensor([-128, -1, 0, 127], dtype=torch.int8).numpy()
    other = torch.tensor([-127, 1, -1, 126], dtype=torch.int8).numpy()
    out = torch.empty(4, dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.min_3"] = min_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.min_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.min_3'.")


check_valid('torch.min', generated_inputs['torch.min_3'], lib="torch", suffix=3)
