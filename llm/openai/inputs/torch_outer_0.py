
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def outer_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.arange(1., 5., dtype=torch.float32).numpy()
    vec2_arr = torch.arange(1., 4., dtype=torch.float32).numpy()
    out_arr = torch.empty((4, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec2": vec2_arr, "out": out_arr}))

    # 2
    input_arr = torch.tensor([1, 0, -2, 7], dtype=torch.int64).numpy()
    vec2_arr = torch.tensor([-3, 2], dtype=torch.int64).numpy()
    out_arr = torch.zeros((4, 2), dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec2": vec2_arr, "out": out_arr}))

    # 3
    input_arr = torch.tensor([-10, 20, 30], dtype=torch.int32).numpy()
    vec2_arr = torch.tensor([0.5, -1.5, 2.0, 3.5], dtype=torch.float64).numpy()
    out_arr = torch.empty((3, 4), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec2": vec2_arr, "out": out_arr}))

    # 4
    input_arr = torch.tensor([1+2j, -3+0.5j], dtype=torch.complex64).numpy()
    vec2_arr = torch.tensor([0-1j, 2+2j, 3-3j], dtype=torch.complex64).numpy()
    out_arr = torch.zeros((2, 3), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec2": vec2_arr, "out": out_arr}))

    # 5
    input_arr = torch.tensor([1.5, -2.5, 3.0], dtype=torch.float16).numpy()
    vec2_arr = torch.tensor([4.0], dtype=torch.float16).numpy()
    out_arr = torch.zeros((3, 1), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec2": vec2_arr, "out": out_arr}))

    # 6
    input_arr = torch.tensor([1-1j], dtype=torch.complex128).numpy()
    vec2_arr = torch.tensor([5+0j, -2+4j], dtype=torch.complex128).numpy()
    out_arr = torch.empty((1, 2), dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec2": vec2_arr, "out": out_arr}))

    # 7
    input_arr = torch.empty((0,), dtype=torch.float32).numpy()
    vec2_arr = torch.arange(0., 5., dtype=torch.float32).numpy()
    out_arr = torch.empty((0, 5), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec2": vec2_arr, "out": out_arr}))

    # 8
    input_arr = torch.empty((0,), dtype=torch.float64).numpy()
    vec2_arr = torch.empty((0,), dtype=torch.float64).numpy()
    out_arr = torch.zeros((0, 0), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec2": vec2_arr, "out": out_arr}))

    # 9
    input_arr = torch.tensor([1000, -2000], dtype=torch.int16).numpy()
    vec2_arr = torch.tensor([300, -400, 500], dtype=torch.int16).numpy()
    out_arr = torch.empty((2, 3), dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec2": vec2_arr, "out": out_arr}))

    # 10
    input_arr = torch.arange(0., 10., dtype=torch.float64).numpy()[::2]
    vec2_arr = torch.arange(1., 9., dtype=torch.float64).numpy()[1::2]
    out_arr = torch.empty((input_arr.shape[0], vec2_arr.shape[0]), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec2": vec2_arr, "out": out_arr}))

    # 11
    input_arr = torch.tensor([0, 1, 255], dtype=torch.uint8).numpy()
    vec2_arr = torch.tensor([2, 3], dtype=torch.uint8).numpy()
    out_arr = torch.ones((3, 2), dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec2": vec2_arr, "out": out_arr}))

    # 12
    input_arr = torch.tensor([-1, -2, -3, -4], dtype=torch.int8).numpy()
    vec2_arr = torch.tensor([10], dtype=torch.int8).numpy()
    out_arr = torch.empty((4, 1), dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec2": vec2_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.outer"] = outer_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.outer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.outer'.")


check_valid('torch.outer', generated_inputs['torch.outer'], lib="torch", suffix=0)
