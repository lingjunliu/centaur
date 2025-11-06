
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def hardsigmoid_inputs():
    list_of_inputs = []

    arr = np.array([-4.0, -3.0, -2.5, 0.0, 2.5, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": arr, "inplace": False}))

    arr = np.random.randn(2, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "inplace": True}))

    arr = np.array(5.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": arr, "inplace": False}))

    arr = np.linspace(-6, 6, 24, dtype=np.float32).reshape(2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"input": arr, "inplace": True}))

    arr = np.array([[-1000.0, -3.0, 0.0, 3.0, 1000.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": arr, "inplace": False}))

    arr = np.random.uniform(-5, 5, (2, 3, 8, 8)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"input": arr, "inplace": False}))

    arr = np.arange(12, dtype=np.float32).reshape(3, 4).T
    list_of_inputs.append(copy.deepcopy({"input": arr, "inplace": False}))

    arr = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": arr, "inplace": False}))

    arr = np.array([-np.inf, -10.0, np.nan, 0.0, 10.0, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": arr, "inplace": False}))

    arr = np.empty((2, 0, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": arr, "inplace": False}))

    arr = np.random.randn(5, 1, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "inplace": True}))

    return list_of_inputs

generated_inputs["torch.nn.functional.hardsigmoid"] = hardsigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.hardsigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hardsigmoid'.")


check_valid('torch.nn.functional.hardsigmoid', generated_inputs['torch.nn.functional.hardsigmoid'], lib="torch", suffix=0)
