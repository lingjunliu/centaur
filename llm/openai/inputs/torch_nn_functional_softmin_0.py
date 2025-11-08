
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def softmin_inputs():
    list_of_inputs = []

    input_arr = np.array([1.0, -2.0, 3.5, 0.0], dtype=np.float32)
    input_dict = {"input": input_arr, "dim": 0, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(3, 4).astype(np.float64)
    input_dict = {"input": input_arr, "dim": 1, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"input": input_arr, "dim": 0, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.linspace(-5, 5, 24, dtype=np.float32).reshape(2, 3, 4)
    input_dict = {"input": input_arr, "dim": -1, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.ones((2, 1, 3, 5), dtype=np.float64)
    input_dict = {"input": input_arr, "dim": -2, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[1000.0, -1000.0, 0.0]], dtype=np.float32)
    input_dict = {"input": input_arr, "dim": -1, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.arange(5, dtype=np.float32)
    input_dict = {"input": input_arr, "dim": 0, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(4, 2, 3).astype(np.float64)
    input_dict = {"input": input_arr, "dim": -2, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(2, 2, 2, 2, 3).astype(np.float32)
    input_dict = {"input": input_arr, "dim": 4, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([-10.0, -10.0, -10.0, 0.0, 10.0], dtype=np.float64)
    input_dict = {"input": input_arr, "dim": 0, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    input_dict = {"input": input_arr, "dim": 0, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (np.random.rand(2, 5, 3).astype(np.float32) * 10) - 5
    input_dict = {"input": input_arr, "dim": 1, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.softmin"] = softmin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.softmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.softmin'.")


check_valid('torch.nn.functional.softmin', generated_inputs['torch.nn.functional.softmin'], lib="torch", suffix=0)
