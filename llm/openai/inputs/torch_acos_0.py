
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def acos_inputs():
    list_of_inputs = []

    input = np.array([0.25, -0.5, 1.0, -1.0], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[0.0, 0.5, -0.5],
                      [0.8, -0.8, 1.0]], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array(0.123456789, dtype=np.float64)
    out = np.empty((), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[[ -0.1, 0.0, 0.1],
                       [ 0.2, -0.2, 0.3]],
                      [[ -0.3, 0.4, -0.4],
                       [ 0.5, -0.5, 0.6]]], dtype=np.float16)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[True, False],
                      [False, True]], dtype=bool)
    out = np.empty(input.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([-1, 0, 1, 1, -1], dtype=np.int8)
    out = np.empty(input.shape, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.asfortranarray(np.array([[-0.2, 0.2],
                                        [ 0.7, -0.7]], dtype=np.float32))
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.linspace(-1.0, 1.0, num=1*2*3*4, dtype=np.float32).reshape(1, 2, 3, 4)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([-1.0, 0.0, 1.0, np.nan], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    input = base[:, ::2]
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([-1.5, -1.1, 1.2, 2.0], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.acos"] = acos_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.acos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.acos'.")


check_valid('torch.acos', generated_inputs['torch.acos'], lib="torch", suffix=0)
