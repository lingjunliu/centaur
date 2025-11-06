
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logaddexp2_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float32).numpy()
    other_arr = torch.tensor([0.5, -2.5, 3.0], dtype=torch.float32).numpy()
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 2
    input_arr = torch.tensor([[1.0, -2.0, 3.5], [4.2, 0.0, -1.1]], dtype=torch.float64).numpy()
    other_arr = torch.tensor([[0.1, 2.2, -3.3], [-4.4, 5.5, 6.6]], dtype=torch.float64).numpy()
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 3 scalar vs vector
    input_arr = torch.tensor(3.0, dtype=torch.float32).numpy()
    other_arr = torch.tensor([1.0, -1.0, 0.0, 10.0], dtype=torch.float32).numpy()
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=other_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 4 broadcast (1,3) with (2,1,3)
    input_arr = torch.tensor([[1.0, -5.0, 2.0]], dtype=torch.float32).numpy()
    other_arr = torch.tensor([[[0.0, 2.0, -2.0]], [[3.0, -1.0, 1.5]]], dtype=torch.float32).numpy()
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=other_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 5 vector and length-1 vector (broadcast)
    input_arr = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=torch.float64).numpy()
    other_arr = torch.tensor([-10.0], dtype=torch.float64).numpy()
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 6 float16 2D
    input_arr = torch.tensor([[1.0, 2.0, -3.0, 4.0],
                              [-1.0, -2.0, 3.0, -4.0],
                              [0.5, -0.5, 1.5, -1.5],
                              [10.0, -10.0, 0.0, 0.25]], dtype=torch.float16).numpy()
    other_arr = torch.tensor([[0.1, -0.2, 0.3, -0.4],
                              [4.0, 5.0, -6.0, 7.0],
                              [1.0, 2.0, 3.0, 4.0],
                              [8.0, 7.0, -6.0, -5.0]], dtype=torch.float16).numpy()
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 7 non-contiguous via slicing
    input_arr = torch.arange(0, 6, dtype=torch.float32)[::2].numpy()
    other_arr = torch.tensor([3.0, -3.0, 1.5], dtype=torch.float32).numpy()[::-1]
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 8 non-contiguous via transpose
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    input_arr = base.numpy()
    other_arr = torch.full_like(base, 1.5).numpy()
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 9 include infs
    input_arr = torch.tensor([float('inf'), -float('inf'), 0.0], dtype=torch.float64).numpy()
    other_arr = torch.tensor([-1.0, 1.0, float('inf')], dtype=torch.float64).numpy()
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 10 empty 1D
    input_arr = torch.empty(0, dtype=torch.float32).numpy()
    other_arr = torch.empty(0, dtype=torch.float32).numpy()
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 11 high-dim broadcast
    input_arr = torch.randn(2, 1, 3, 1, dtype=torch.float32).numpy()
    other_arr = torch.randn(1, 1, 1, 5, dtype=torch.float32).numpy()
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 12 zero-size inner dim
    input_arr = torch.empty(2, 0, 3, dtype=torch.float64).numpy()
    other_arr = torch.empty(1, 0, 1, dtype=torch.float64).numpy()
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.logaddexp2"] = logaddexp2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logaddexp2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logaddexp2'.")


check_valid('torch.logaddexp2', generated_inputs['torch.logaddexp2'], lib="torch", suffix=0)
