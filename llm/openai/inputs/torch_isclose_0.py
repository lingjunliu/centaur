
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def isclose_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other = torch.tensor([1.0 + 1e-10, 3.0, 4.0]).numpy()
    input_dict = {"input": input, "other": other, "rtol": 1e-05, "atol": 1e-08, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([float('inf'), 4.0]).numpy()
    other = torch.tensor([float('inf'), 6.0]).numpy()
    input_dict = {"input": input, "other": other, "rtol": 0.5, "atol": 1e-08, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]], dtype=torch.float32).numpy()
    other = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]], dtype=torch.float32).numpy()
    input_dict = {"input": input, "other": other, "rtol": 0.0, "atol": 0.0, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor(1.0).numpy()
    other = torch.tensor(1.000009).numpy()
    input_dict = {"input": input, "other": other, "rtol": 1e-05, "atol": 0.0, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    input = t.numpy()
    other = (t + 1e-4).numpy()
    input_dict = {"input": input, "other": other, "rtol": 1e-03, "atol": 1e-05, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32).numpy()
    other = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).numpy()
    input_dict = {"input": input, "other": other, "rtol": 1e-05, "atol": 1e-08, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([float('nan'), 1.0], dtype=torch.float32).numpy()
    other = torch.tensor([float('nan'), 1.0000001], dtype=torch.float32).numpy()
    input_dict = {"input": input, "other": other, "rtol": 1e-06, "atol": 1e-08, "equal_nan": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([float('nan'), 2.0], dtype=torch.float32).numpy()
    other = torch.tensor([float('nan'), 2.0], dtype=torch.float32).numpy()
    input_dict = {"input": input, "other": other, "rtol": 0.0, "atol": 0.0, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([0.1, 0.2, 0.3, -0.1, -0.2], dtype=torch.float16).numpy()
    other = torch.tensor([0.1005, 0.1995, 0.3004, -0.1004, -0.2006], dtype=torch.float16).numpy()
    input_dict = {"input": input, "other": other, "rtol": 1e-02, "atol": 1e-03, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([1e10, -1e-10], dtype=torch.float64).numpy()
    other = torch.tensor([1e10 + 1e5, 0.0], dtype=torch.float64).numpy()
    input_dict = {"input": input, "other": other, "rtol": 1e-04, "atol": 1e-10, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([-0.0, 0.0], dtype=torch.float32).numpy()
    other = torch.tensor([0.0, -0.0], dtype=torch.float32).numpy()
    input_dict = {"input": input, "other": other, "rtol": 0.0, "atol": 0.0, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=torch.float32).numpy()
    other = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    input_dict = {"input": input, "other": other, "rtol": 1e-05, "atol": 0.0, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32).numpy()
    other = torch.tensor([1.0, -2.0, 3.1, -4.0], dtype=torch.float32).numpy()
    input_dict = {"input": input, "other": other, "rtol": 0.0, "atol": 0.2, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros((2, 2, 2, 2), dtype=torch.float32).numpy()
    other = torch.full((2, 2, 2, 2), 1e-7, dtype=torch.float32).numpy()
    input_dict = {"input": input, "other": other, "rtol": 0.0, "atol": 1e-6, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.isclose"] = isclose_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.isclose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isclose'.")


check_valid('torch.isclose', generated_inputs['torch.isclose'], lib="torch", suffix=0)
