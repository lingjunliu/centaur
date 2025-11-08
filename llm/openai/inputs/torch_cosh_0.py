
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def cosh_inputs():
    list_of_inputs = []

    # 1: 1D float32
    input = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 2: 2D float64 with large magnitudes
    input = np.array([[0.0, 1000.0], [-1000.0, 5.0]], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 3: 3D int32
    input = np.array([[[1, -2, 3], [4, -5, 6]],
                      [[-7, 8, -9], [10, -11, 12]]], dtype=np.int32)
    out = np.empty(input.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 4: 0-dim scalar float32
    input = np.array(-2.5, dtype=np.float32)
    out = np.empty((), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 5: boolean tensor
    input = np.array([[True, False, True], [False, True, False]], dtype=bool)
    out = np.empty(input.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 6: complex64 1D
    input = np.array([1+1j, -1-2j, 0+0j, 2-1j], dtype=np.complex64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 7: complex128 2D
    input = np.array([[1j, -2j], [3+4j, -5-6j]], dtype=np.complex128)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 8: float16 1D
    input = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float16)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 9: non-contiguous transpose float64
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    input = base.T
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 10: 4D float32
    input = np.linspace(-2.0, 2.0, num=2*3*4*5, dtype=np.float32).reshape(2, 3, 4, 5)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 11: empty tensor shape (0, 3)
    input = np.empty((0, 3), dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 12: reversed slice (non-contiguous) float64
    base = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    input = base[::-1]
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.cosh"] = cosh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cosh'.")


check_valid('torch.cosh', generated_inputs['torch.cosh'], lib="torch", suffix=0)
