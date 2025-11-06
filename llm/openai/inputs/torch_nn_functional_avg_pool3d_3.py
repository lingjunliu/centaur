
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def avg_pool3d_inputs_3():
    list_of_inputs = []

    # Input 1
    input_arr = np.arange(1*1*4*4*4, dtype=np.float32).reshape(1,1,4,4,4) - 10.0
    ks = [2, 2, 2]
    st = [2, 2, 2]
    pd = [0, 0, 0]
    input_dict = {
        "input": input_arr,
        "kernel_size": ks,
        "stride": st,
        "padding": pd,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = np.random.randn(2,3,5,6,7).astype(np.float32)
    ks = [3, 3, 3]
    st = [1, 1, 1]
    pd = [1, 1, 1]
    input_dict = {
        "input": input_arr,
        "kernel_size": ks,
        "stride": st,
        "padding": pd,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 27
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = (np.random.randn(2,3,5,6,7)*2 - 1).astype(np.float32)
    ks = [2, 3, 4]
    st = [2, 3, 4]
    pd = [0, 1, 2]
    input_dict = {
        "input": input_arr,
        "kernel_size": ks,
        "stride": st,
        "padding": pd,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 24
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = (np.random.randn(1,2,3,5,7)).astype(np.float64)
    ks = [1, 2, 3]
    st = [1, 2, 3]
    pd = [0, 0, 0]
    input_dict = {
        "input": input_arr,
        "kernel_size": ks,
        "stride": st,
        "padding": pd,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = np.linspace(-3, 3, num=1*1*3*3*3, dtype=np.float32).reshape(1,1,3,3,3)
    ks = [3, 3, 3]
    st = [1, 1, 1]
    pd = [1, 1, 1]
    input_dict = {
        "input": input_arr,
        "kernel_size": ks,
        "stride": st,
        "padding": pd,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 27
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = (np.random.randn(1,1,8,8,8)).astype(np.float32)
    ks = [5, 5, 5]
    st = [2, 2, 2]
    pd = [0, 0, 0]
    input_dict = {
        "input": input_arr,
        "kernel_size": ks,
        "stride": st,
        "padding": pd,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 13
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = (np.random.randn(3,1,6,4,5)).astype(np.float32)
    ks = [2, 1, 3]
    st = [2, 1, 2]
    pd = [0, 0, 1]
    input_dict = {
        "input": input_arr,
        "kernel_size": ks,
        "stride": st,
        "padding": pd,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = (np.random.randn(2,2,4,4,4)).astype(np.float32)
    ks = [2, 2, 2]
    st = [2, 2, 2]
    pd = [0, 0, 0]
    input_dict = {
        "input": input_arr,
        "kernel_size": ks,
        "stride": st,
        "padding": pd,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = np.array([[[[[-1.0, 2.0],
                              [3.0, -4.0]],
                             [[5.0, -6.0],
                              [7.0, 8.0]]]]], dtype=np.float32)
    ks = [1, 1, 1]
    st = [1, 1, 1]
    pd = [0, 0, 0]
    input_dict = {
        "input": input_arr,
        "kernel_size": ks,
        "stride": st,
        "padding": pd,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = (np.random.randn(1,4,7,3,2)*5).astype(np.float32)
    ks = [3, 2, 2]
    st = [1, 2, 1]
    pd = [1, 0, 1]
    input_dict = {
        "input": input_arr,
        "kernel_size": ks,
        "stride": st,
        "padding": pd,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 12
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = (np.random.randn(2,4,3,3,3)).astype(np.float32)
    ks = [2, 2, 2]
    st = [1, 1, 1]
    pd = [0, 0, 0]
    input_dict = {
        "input": input_arr,
        "kernel_size": ks,
        "stride": st,
        "padding": pd,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (fixed padding to be <= floor(kernel_size/2))
    input_arr = (np.random.randn(1,2,2,2,2)).astype(np.float32)
    ks = [2, 2, 2]
    st = [2, 2, 2]
    pd = [1, 1, 1]
    input_dict = {
        "input": input_arr,
        "kernel_size": ks,
        "stride": st,
        "padding": pd,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.avg_pool3d_3"] = avg_pool3d_inputs_3()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.avg_pool3d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool3d_3'.")


check_valid('torch.nn.functional.avg_pool3d', generated_inputs['torch.nn.functional.avg_pool3d_3'], lib="torch", suffix=3)
