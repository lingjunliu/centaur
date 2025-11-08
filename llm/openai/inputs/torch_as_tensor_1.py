
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def as_tensor_inputs():
    list_of_inputs = []

    data = [np.int32(1), np.int32(2), np.int32(3)]
    dtype = torch.int64
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = [np.float64(-1.5), np.float64(2.5), np.float64(-3.0)]
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = [
        [np.int8(1), np.int8(-2)],
        [np.int8(3), np.int8(-4)]
    ]
    dtype = torch.int16
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = [
        [np.bool_(True), np.bool_(False)],
        [np.bool_(False), np.bool_(True)]
    ]
    dtype = torch.bool
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = [np.uint8(0), np.uint8(255), np.uint8(128), np.uint8(64), np.uint8(1), np.uint8(2)]
    dtype = torch.uint8
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = [np.complex128(1 + 2j), np.complex128(-3 + 0.5j)]
    dtype = torch.complex64
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = [
        [
            [np.float16(0.0), np.float16(0.5), np.float16(1.0)],
            [np.float16(-0.5), np.float16(2.0), np.float16(-1.5)]
        ],
        [
            [np.float16(3.0), np.float16(-2.0), np.float16(4.5)],
            [np.float16(6.0), np.float16(-3.5), np.float16(0.25)]
        ]
    ]
    dtype = torch.float16
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = []
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = [
        [
            [np.float32(0.0), np.float32(1.0)],
            [np.float32(2.0), np.float32(3.0)]
        ],
        [
            [np.float32(-1.0), np.float32(-2.0)],
            [np.float32(4.0), np.float32(5.0)]
        ]
    ]
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = [np.float32(3.14159)]
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = [np.int64(-(2**40)), np.int64(2**40 - 1)]
    dtype = torch.int64
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = [
        [np.complex128(0 + 0j), np.complex128(-1 + 2j), np.complex128(3 - 4j)],
        [np.complex128(5 + 0j), np.complex128(-6 - 1j), np.complex128(7 + 8j)]
    ]
    dtype = torch.complex128
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    return list_of_inputs

generated_inputs["torch.as_tensor_1"] = as_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.as_tensor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.as_tensor_1'.")


check_valid('torch.as_tensor', generated_inputs['torch.as_tensor_1'], lib="torch", suffix=1)
