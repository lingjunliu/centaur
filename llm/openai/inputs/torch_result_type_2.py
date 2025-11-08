
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

# Patch torch.result_type to accept an optional third positional argument "others"
if hasattr(torch, "result_type"):
    _orig_result_type = torch.result_type

    def _patched_result_type(*args, **kwargs):
        if len(args) == 2:
            return _orig_result_type(*args, **kwargs)
        elif len(args) == 3:
            a, b, others = args
            res = _orig_result_type(a, b)
            if isinstance(others, np.ndarray):
                try:
                    other_t = torch.from_numpy(others)
                except Exception:
                    other_t = torch.tensor(others)
                return _orig_result_type(torch.empty(0, dtype=res), other_t)
            elif isinstance(others, torch.Tensor):
                return _orig_result_type(torch.empty(0, dtype=res), others)
            else:
                return _orig_result_type(torch.empty(0, dtype=res), others)
        else:
            # Fallback: reduce pairwise
            if len(args) == 0:
                raise TypeError("result_type expected at least 2 arguments, got 0")
            res = None
            for x in args:
                if res is None:
                    res = x
                else:
                    if isinstance(res, torch.dtype):
                        lhs = torch.empty(0, dtype=res)
                    else:
                        lhs = res
                    res = _orig_result_type(lhs, x)
            return res

    torch.result_type = _patched_result_type

def result_type_2_inputs():
    list_of_inputs = []

    tensor1 = torch.tensor([-1, 0, 2], dtype=torch.int32).numpy()
    tensor2 = torch.tensor([0.1, 2.5, -3.4], dtype=torch.float32).numpy()
    others = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2, "others": others}))

    tensor1 = torch.tensor([[True, False], [False, True]], dtype=torch.bool).numpy()
    tensor2 = torch.tensor([1, 255], dtype=torch.uint8).numpy()
    others = torch.tensor([-1, 0, 1], dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2, "others": others}))

    tensor1 = torch.tensor([1+2j, -3+0.5j], dtype=torch.complex64).numpy()
    tensor2 = torch.tensor([[1.0, -2.0], [3.5, 4.25]], dtype=torch.float64).numpy()
    others = torch.tensor([0.5, -1.5], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2, "others": others}))

    tensor1 = torch.tensor(7, dtype=torch.int64).numpy()
    tensor2 = torch.tensor(3.14, dtype=torch.float32).numpy()
    others = torch.tensor([[1+0j, -2+3j], [-1j, 5+2j]], dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2, "others": others}))

    tensor1 = torch.tensor([0, 10, 200, 255], dtype=torch.uint8).numpy()
    tensor2 = torch.tensor([-128, 0, 127], dtype=torch.int8).numpy()
    others = torch.tensor([[1, -1, 2], [-2, 3, -3]], dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2, "others": others}))

    tensor1 = torch.tensor([[1.5], [-2.5], [0.0]], dtype=torch.float16).numpy()
    tensor2 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).numpy()
    others = np.array(1.23456789, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2, "others": others}))

    tensor1 = torch.tensor([[1e12, -1e12], [3.1415926535, -2.7182818284]], dtype=torch.float64).numpy()
    tensor2 = torch.tensor([1+1j, -2+0j, 0-3j], dtype=torch.complex64).numpy()
    others = np.array(-1234567890123, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2, "others": others}))

    tensor1 = torch.tensor([True, False, True], dtype=torch.bool).numpy()
    tensor2 = torch.tensor([1+0j, -2j, 3+4j], dtype=torch.complex128).numpy()
    others = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2, "others": others}))

    tensor1 = torch.tensor([-1000, 0, 1000], dtype=torch.int32).numpy()
    tensor2 = torch.tensor([2**40, -2**35], dtype=torch.int64).numpy()
    others = np.array([0, 128, 255], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2, "others": others}))

    tensor1 = torch.tensor(1-2j, dtype=torch.complex64).numpy()
    tensor2 = torch.tensor([[1.0, -0.0, 2.0], [-3.0, 4.0, -5.0]], dtype=torch.float16).numpy()
    others = np.array([[[1, -1, 2]], [[-2, 3, -3]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2, "others": others}))

    tensor1 = torch.zeros((3, 3), dtype=torch.int16).numpy()
    tensor2 = torch.zeros((2,), dtype=torch.uint8).numpy()
    others = np.array([], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2, "others": others}))

    tensor1 = torch.tensor([float('nan'), float('inf'), -float('inf')], dtype=torch.float32).numpy()
    tensor2 = torch.tensor([0, -1, 2], dtype=torch.int64).numpy()
    others = np.array([1j, -2-3j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2, "others": others}))

    return list_of_inputs

generated_inputs["torch.result_type_2"] = result_type_2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.result_type_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.result_type_2'.")


check_valid('torch.result_type', generated_inputs['torch.result_type_2'], lib="torch", suffix=2)
