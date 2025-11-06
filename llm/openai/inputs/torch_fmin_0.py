
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def fmin_inputs():
    list_of_inputs = []

    # 1) 1D float32 with NaNs
    input = torch.tensor([2.2, float('nan'), 2.1, float('nan')], dtype=torch.float32).numpy()
    other = torch.tensor([-9.3, 0.1, float('nan'), float('nan')], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 2) 1D int64 with negatives
    input = torch.tensor([-10, 0, 5, 7, -3], dtype=torch.int64).numpy()
    other = torch.tensor([3, -2, 6, -8, -3], dtype=torch.int64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 3) 2D broadcasting float64
    input = torch.tensor([[1.0, -2.0, 3.5],
                          [4.2, 5.1, -6.0]], dtype=torch.float64).numpy()
    other = torch.tensor([[0.0, -3.0, 2.0]], dtype=torch.float64).numpy()
    out_shape = np.broadcast_shapes(input.shape, other.shape)
    out = np.empty(out_shape, dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 4) 0D scalar float32 with NaN vs finite
    input = torch.tensor(float('nan'), dtype=torch.float32).numpy()
    other = torch.tensor(3.5, dtype=torch.float32).numpy()
    out = np.empty((), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 5) 3D broadcasting float32
    input = torch.randn((3, 1, 4), dtype=torch.float32).numpy()
    other = torch.randn((1, 5, 1), dtype=torch.float32).numpy()
    out_shape = np.broadcast_shapes(input.shape, other.shape)
    out = np.empty(out_shape, dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 6) uint8 broadcasting
    input = torch.tensor([[255, 10],
                          [0, 128]], dtype=torch.uint8).numpy()
    other = torch.tensor([[100],
                          [200]], dtype=torch.uint8).numpy()
    out_shape = np.broadcast_shapes(input.shape, other.shape)
    out = np.empty(out_shape, dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 7) 1D int32 with scalar other
    input = torch.tensor([10, -5, 0, 7], dtype=torch.int32).numpy()
    other = torch.tensor(-1, dtype=torch.int32).numpy()
    out_shape = np.broadcast_shapes(input.shape, other.shape)
    out = np.empty(out_shape, dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 8) 4D broadcasting float16
    input = torch.randn((1, 2, 1, 3), dtype=torch.float16).numpy()
    other = torch.randn((4, 1, 5, 1), dtype=torch.float16).numpy()
    out_shape = np.broadcast_shapes(input.shape, other.shape)
    out = np.empty(out_shape, dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 9) float64 with inf, -inf, NaN
    input = torch.tensor([float('inf'), float('-inf'), 0.0], dtype=torch.float64).numpy()
    other = torch.tensor([1.0, 1.0, float('nan')], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 10) int16 broadcasting (5,1) and scalar (1,)
    input = torch.tensor([[5], [3], [-2], [7], [0]], dtype=torch.int16).numpy()
    other = torch.tensor([2], dtype=torch.int16).numpy()
    out_shape = np.broadcast_shapes(input.shape, other.shape)
    out = np.empty(out_shape, dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 11) Non-contiguous-like numpy view float32
    base = np.arange(16, dtype=np.float32).reshape(4, 4)
    input = base[::2, ::2]
    other = np.array([[1.0, -1.0],
                      [2.0, -2.0]], dtype=np.float32)
    out_shape = np.broadcast_shapes(input.shape, other.shape)
    out = np.empty(out_shape, dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 12) int8 large broadcast (2,3,4) with (1,1,4)
    input = torch.tensor([[[1, -2, 3, -4],
                           [5, -6, 7, -8],
                           [9, -10, 11, -12]],
                          [[-1, 2, -3, 4],
                           [-5, 6, -7, 8],
                           [-9, 10, -11, 12]]], dtype=torch.int8).numpy()
    other = torch.tensor([[[0, 1, -1, 2]]], dtype=torch.int8).numpy()
    out_shape = np.broadcast_shapes(input.shape, other.shape)
    out = np.empty(out_shape, dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.fmin"] = fmin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fmin'.")


check_valid('torch.fmin', generated_inputs['torch.fmin'], lib="torch", suffix=0)
