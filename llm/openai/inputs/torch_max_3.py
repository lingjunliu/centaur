
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

class IndexableInt(int):
    def __new__(cls, value):
        return int.__new__(cls, int(value))
    def __len__(self):
        return 1
    def __iter__(self):
        yield int(self)
    def __getitem__(self, idx):
        if idx in (0, -1):
            return int(self)
        raise IndexError("Index out of range for IndexableInt")

def torch_max_3_inputs():
    list_of_inputs = []

    def make_case(arr, dim_scalar, keepdim):
        ndim = arr.ndim
        d = int(dim_scalar)
        d_eff = d if d >= 0 else d + ndim
        assert 0 <= d_eff < ndim
        out_shape = list(arr.shape)
        if keepdim:
            out_shape[d_eff] = 1
        else:
            out_shape.pop(d_eff)
        values_out = np.empty(out_shape, dtype=arr.dtype)
        indices_out = np.empty(out_shape, dtype=np.int64)
        return {
            "input": arr,
            "dim": IndexableInt(d),
            "keepdim": keepdim,
            "out": (values_out, indices_out)
        }

    a1 = np.array([[0.5, -1.2, 3.4, 2.2],
                   [4.1, -0.7, 0.0, 5.0],
                   [-3.0, -2.0, -1.0, -4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(make_case(a1, 1, False)))

    a2 = (np.arange(12, dtype=np.float32).reshape(3, 4) - 6.0).astype(np.float32)
    list_of_inputs.append(copy.deepcopy(make_case(a2, -1, True)))

    a3 = np.array([10, -5, 7, 7, -1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy(make_case(a3, 0, True)))

    a4 = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy(make_case(a4, 2, True)))

    a5 = np.random.uniform(-1, 1, (2, 2, 3, 4)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy(make_case(a5, 0, False)))

    a6 = (np.arange(4*1*5, dtype=np.int32).reshape(4, 1, 5) - 10).astype(np.int32)
    list_of_inputs.append(copy.deepcopy(make_case(a6, 1, False)))

    a7 = np.array([[True, False, True, True, False],
                   [False, False, True, False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy(make_case(a7, 0, True)))

    a8 = (np.random.randn(2, 4, 3) * 5 - 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy(make_case(a8, -2, False)))

    a9 = np.array([[[42.0, -3.0, 7.0],
                    [0.0, 0.0, 0.0]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(make_case(a9, 2, False)))

    a10 = (np.arange(2*3*4*2*3, dtype=np.int16).reshape(2, 3, 4, 2, 3) - 50).astype(np.int16)
    list_of_inputs.append(copy.deepcopy(make_case(a10, 3, False)))

    a11 = (np.random.randint(-128, 127, size=(8, 2))).astype(np.int8)
    list_of_inputs.append(copy.deepcopy(make_case(a11, 0, False)))

    a12 = np.linspace(-5, 5, 30, dtype=np.float64).reshape(3, 5, 2)
    list_of_inputs.append(copy.deepcopy(make_case(a12, -3, True)))

    return list_of_inputs

generated_inputs["torch.max_3"] = torch_max_3_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.max_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.max_3'.")


check_valid('torch.max', generated_inputs['torch.max_3'], lib="torch", suffix=3)
