
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_max_pool2d_inputs():
    list_of_inputs = []

    input = np.array([[[[1.0, -2.0, 3.0, -4.0],
                        [5.0, -6.0, 7.0, -8.0],
                        [9.0, -10.0, 11.0, -12.0],
                        [13.0, -14.0, 15.0, -16.0]]]], dtype=np.float32)
    output_size = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.random.randn(2, 3, 7, 5).astype(np.float32)
    output_size = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.random.randn(3, 8, 8).astype(np.float64)
    output_size = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.random.randint(-50, 50, size=(1, 2, 9, 9)).astype(np.float64)
    output_size = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.random.uniform(-1, 1, size=(5, 1, 6, 10)).astype(np.float32)
    output_size = 5
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.arange(25, dtype=np.float32).reshape(1, 5, 5)
    output_size = 5
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = (np.random.randn(10, 3, 2).astype(np.float32) * 10.0)
    output_size = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.random.randn(2, 4, 13, 17).astype(np.float64)
    output_size = 7
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = -np.abs(np.random.randn(3, 2, 3, 3).astype(np.float32)) * 100.0
    output_size = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.random.randn(2, 9, 7).astype(np.float64)
    output_size = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.random.randn(2, 2, 100, 1).astype(np.float64)
    output_size = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.random.randn(4, 32, 5).astype(np.float32)
    output_size = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_max_pool2d_2"] = adaptive_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.adaptive_max_pool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_max_pool2d_2'.")


check_valid('torch.nn.functional.adaptive_max_pool2d', generated_inputs['torch.nn.functional.adaptive_max_pool2d_2'], lib="torch", suffix=2)
