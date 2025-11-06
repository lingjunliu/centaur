
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def fake_quantize_per_tensor_affine_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, -2.5, 3.3], dtype=torch.float32).numpy()
    scale = np.float32(0.1)
    zero_point = np.int32(0)
    quant_min = np.int32(0)
    quant_max = np.int32(255)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max
    }))

    input = torch.tensor([[0.5, -1.2, 3.0],
                          [4.5, -0.7, 2.2]], dtype=torch.float32).numpy()
    scale = np.float64(0.05)
    zero_point = np.int64(128)
    quant_min = np.int64(0)
    quant_max = np.int64(255)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max
    }))

    input = torch.tensor([[[0.1, -0.2],
                           [0.3, -0.4]],
                          [[0.5, -0.6],
                           [0.7, -0.8]]], dtype=torch.float16).numpy()
    scale = np.float16(0.02)
    zero_point = np.int8(0)
    quant_min = np.int8(-128)
    quant_max = np.int8(127)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max
    }))

    input = torch.linspace(-1.0, 1.0, steps=48, dtype=torch.float32).reshape(1, 3, 4, 4).numpy()
    scale = np.float32(0.01)
    zero_point = np.int32(0)
    quant_min = np.int32(-128)
    quant_max = np.int32(127)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max
    }))

    input = torch.tensor(0.75, dtype=torch.float64).numpy()
    scale = np.float64(0.2)
    zero_point = np.int64(10)
    quant_min = np.int64(0)
    quant_max = np.int64(127)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max
    }))

    base = torch.arange(0, 10, dtype=torch.float32).numpy()
    input = base[::2]
    scale = np.float32(0.5)
    zero_point = np.int32(5)
    quant_min = np.int32(0)
    quant_max = np.int32(15)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max
    }))

    input = torch.arange(-12, 12, dtype=torch.float32).reshape(2, 3, 4).numpy()
    scale = np.float32(1.0)
    zero_point = np.int32(0)
    quant_min = np.int32(-10)
    quant_max = np.int32(10)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max
    }))

    input = torch.tensor([0.0, 1.5, -1.5], dtype=torch.float64).numpy()
    scale = np.float64(0.333)
    zero_point = np.int32(3)
    quant_min = np.int32(0)
    quant_max = np.int32(10)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max
    }))

    input = torch.linspace(-0.5, 0.5, steps=25, dtype=torch.float16).reshape(5, 5).numpy()
    scale = np.float16(0.001)
    zero_point = np.int16(0)
    quant_min = np.int16(-32)
    quant_max = np.int16(31)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max
    }))

    input = torch.empty((0,), dtype=torch.float32).numpy()
    scale = np.float32(0.1)
    zero_point = np.int32(0)
    quant_min = np.int32(0)
    quant_max = np.int32(255)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max
    }))

    input = (torch.arange(0, 16, dtype=torch.float32) * 1000.0).reshape(4, 4).numpy()
    scale = np.float32(100.0)
    zero_point = np.int32(100)
    quant_min = np.int32(0)
    quant_max = np.int32(255)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max
    }))

    input = torch.tensor([-0.1, 0.0, 0.1, 0.2], dtype=torch.float32).numpy()
    scale = np.float32(0.123)
    zero_point = np.int32(1)
    quant_min = np.int32(0)
    quant_max = np.int32(1)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max
    }))

    return list_of_inputs

generated_inputs["torch.fake_quantize_per_tensor_affine"] = fake_quantize_per_tensor_affine_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fake_quantize_per_tensor_affine' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fake_quantize_per_tensor_affine'.")


check_valid('torch.fake_quantize_per_tensor_affine', generated_inputs['torch.fake_quantize_per_tensor_affine'], lib="torch", suffix=0)
