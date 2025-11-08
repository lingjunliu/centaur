
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def floor_divide_inputs():
    list_of_inputs = []

    # 1: 1D float32
    t = torch.tensor([1.0, 2.0, 3.0, -4.5], dtype=torch.float32)
    input_arr = t.numpy()
    other = 2.0
    out = torch.empty_like(t, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 2: 2D int64 -> result float64
    t = torch.tensor([[-10, 10], [3, -3]], dtype=torch.int64)
    input_arr = t.numpy()
    other = 3.0
    out = torch.empty_like(t, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 3: 3D int32 -> result float32
    t = torch.tensor([[[-5, 8], [15, -16]], [[23, -42], [0, 7]]], dtype=torch.int32)
    input_arr = t.numpy()
    other = -2.0
    out = torch.empty_like(t, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 4: 0-D scalar float64
    t = torch.tensor(3.9, dtype=torch.float64)
    input_arr = t.numpy()
    other = -1.5
    out = torch.empty_like(t, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 5: 4D float16
    t = torch.tensor([[[[1.0, -2.0, 3.5]], [[-4.1, 5.2, -6.3]]]], dtype=torch.float16)
    input_arr = t.numpy()
    other = 1.1
    out = torch.empty_like(t, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 6: 1D uint8 -> result float32
    t = torch.tensor([0, 1, 255], dtype=torch.uint8)
    input_arr = t.numpy()
    other = 5.0
    out = torch.empty_like(t, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 7: 2D int8 -> result float32
    t = torch.tensor([[-128, -1, 0], [1, 64, 127]], dtype=torch.int8)
    input_arr = t.numpy()
    other = 1.5
    out = torch.empty_like(t, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 8: Empty 1D float32
    t = torch.tensor([], dtype=torch.float32)
    input_arr = t.numpy()
    other = 2.0
    out = torch.empty_like(t, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 9: Non-contiguous (transpose) int32 -> result float32
    base = torch.arange(12, dtype=torch.int32).view(3, 4)
    t = base.t()  # shape (4, 3), non-contiguous
    input_arr = t.numpy()
    other = -3.5
    out = torch.empty_like(t, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 10: 3D empty dimension int16 -> result float32
    t = torch.empty((2, 0, 3), dtype=torch.int16)
    input_arr = t.numpy()
    other = 7.0
    out = torch.empty_like(t, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 11: Large magnitude int64 -> result float64
    t = torch.tensor([2**62, -(2**62), 1234567890123456789], dtype=torch.int64)
    input_arr = t.numpy()
    other = -2.0
    out = torch.empty_like(t, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 12: 3D float32 with negatives
    t = torch.tensor([[[-1.2, 2.7, -3.8], [4.9, -5.1, 6.6]]], dtype=torch.float32)
    input_arr = t.numpy()
    other = 2.3
    out = torch.empty_like(t, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.floor_divide_2"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.floor_divide_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor_divide_2'.")


check_valid('torch.floor_divide', generated_inputs['torch.floor_divide_2'], lib="torch", suffix=2)
