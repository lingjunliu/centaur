
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def lerp_inputs():
    def make_out(a, b, c):
        shape = np.broadcast(a, b, c).shape
        dtype = np.result_type(a.dtype, b.dtype, c.dtype)
        return np.empty(shape, dtype=dtype)

    list_of_inputs = []

    # 1
    input_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    end = np.full(4, 10.0, dtype=np.float32)
    weight = np.array(0.5, dtype=np.float32)
    out = make_out(input_arr, end, weight)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "end": end, "weight": weight, "out": out}))

    # 2
    input_arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    end = np.ones((2, 3), dtype=np.float64) * 5.0
    weight = np.array([[0.2, 0.4, 0.6],
                       [0.8, 1.0, 1.2]], dtype=np.float64)
    out = make_out(input_arr, end, weight)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "end": end, "weight": weight, "out": out}))

    # 3
    input_arr = np.arange(3, dtype=np.float32).reshape(3, 1)
    end = np.arange(4, dtype=np.float32).reshape(1, 4)
    weight = np.array(0.3, dtype=np.float32)
    out = make_out(input_arr, end, weight)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "end": end, "weight": weight, "out": out}))

    # 4
    input_arr = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    end = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    weight = np.array([-0.5, 0.0, 1.5], dtype=np.float64)
    out = make_out(input_arr, end, weight)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "end": end, "weight": weight, "out": out}))

    # 5
    input_arr = np.linspace(-1, 1, 24, dtype=np.float32).reshape(2, 3, 4)
    end = np.zeros((2, 3, 4), dtype=np.float32)
    weight = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    out = make_out(input_arr, end, weight)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "end": end, "weight": weight, "out": out}))

    # 6
    input_arr = np.array([1.0], dtype=np.float16)
    end = np.array([2.0], dtype=np.float16)
    weight = np.array([0.25], dtype=np.float16)
    out = make_out(input_arr, end, weight)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "end": end, "weight": weight, "out": out}))

    # 7
    input_arr = np.empty((0, 3), dtype=np.float32)
    end = np.empty((0, 3), dtype=np.float32)
    weight = np.array(0.7, dtype=np.float32)
    out = make_out(input_arr, end, weight)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "end": end, "weight": weight, "out": out}))

    # 8
    input_arr = np.array([np.nan, -np.inf], dtype=np.float32)
    end = np.array([np.inf, 3.0], dtype=np.float32)
    weight = np.array([0.1, 0.9], dtype=np.float32)
    out = make_out(input_arr, end, weight)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "end": end, "weight": weight, "out": out}))

    # 9
    input_arr = np.arange(2*1*1, dtype=np.float64).reshape(2, 1, 1) + 1.0
    end = np.ones((1, 3, 4), dtype=np.float64) * 10.0
    weight = np.linspace(0.0, 1.0, 8, dtype=np.float64).reshape(2, 1, 4)
    out = make_out(input_arr, end, weight)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "end": end, "weight": weight, "out": out}))

    # 10
    input_arr = np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    end = np.asfortranarray(np.full((2, 3), 2.0, dtype=np.float32))
    weight = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    out = make_out(input_arr, end, weight)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "end": end, "weight": weight, "out": out}))

    # 11
    input_arr = np.array(3.0, dtype=np.float64)
    end = np.array(7.0, dtype=np.float64)
    weight = np.array(0.25, dtype=np.float64)
    out = make_out(input_arr, end, weight)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "end": end, "weight": weight, "out": out}))

    # 12
    input_arr = np.zeros((1, 0, 4), dtype=np.float32)
    end = np.ones((1, 0, 4), dtype=np.float32)
    weight = np.array(2.0, dtype=np.float32)
    out = make_out(input_arr, end, weight)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "end": end, "weight": weight, "out": out}))

    return list_of_inputs

generated_inputs["torch.lerp_1"] = lerp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.lerp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lerp_1'.")


check_valid('torch.lerp', generated_inputs['torch.lerp_1'], lib="torch", suffix=1)
