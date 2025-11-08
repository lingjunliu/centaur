
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def empty_like_inputs():
    list_of_inputs = []

    input = np.array([1.0, -2.5, 3.3], dtype=np.float32)
    dtype = torch.float32
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype, "requires_grad": requires_grad}))

    input = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int64)
    dtype = torch.int64
    requires_grad = False
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype, "requires_grad": requires_grad}))

    input = np.random.randint(0, 255, size=(2, 3, 4), dtype=np.uint8)
    dtype = torch.uint8
    requires_grad = False
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype, "requires_grad": requires_grad}))

    input = np.array(3.1415926535, dtype=np.float64)
    dtype = torch.float64
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype, "requires_grad": requires_grad}))

    input = np.array([], dtype=np.float32)
    dtype = torch.float32
    requires_grad = False
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype, "requires_grad": requires_grad}))

    input = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    dtype = torch.bool
    requires_grad = False
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype, "requires_grad": requires_grad}))

    base = np.arange(24, dtype=np.int32).reshape(4, 6)
    input = base[:, ::-2]
    dtype = torch.float32
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype, "requires_grad": requires_grad}))

    input = np.array([1+2j, -3+0.5j, 0-1j], dtype=np.complex64)
    dtype = torch.complex64
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype, "requires_grad": requires_grad}))

    input = np.array([[1-1j, 2+0j], [-4+5j, 0-3j]], dtype=np.complex128)
    dtype = torch.complex128
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype, "requires_grad": requires_grad}))

    input = np.random.randn(2, 1, 3, 4).astype(np.float16)
    dtype = torch.float16
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype, "requires_grad": requires_grad}))

    input = np.empty((2, 0, 3), dtype=np.int32)
    dtype = torch.int32
    requires_grad = False
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype, "requires_grad": requires_grad}))

    input = np.arange(24, dtype=np.float64).reshape(1, 2, 3, 1, 4) - 12.5
    dtype = torch.float64
    requires_grad = False
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype, "requires_grad": requires_grad}))

    return list_of_inputs

generated_inputs["torch.empty_like"] = empty_like_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.empty_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_like'.")


check_valid('torch.empty_like', generated_inputs['torch.empty_like'], lib="torch", suffix=0)
