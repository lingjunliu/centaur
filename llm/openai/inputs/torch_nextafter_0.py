
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def nextafter_inputs():
    list_of_inputs = []

    # 1
    input = torch.tensor([1.0, 2.0], dtype=torch.float32).numpy()
    other = torch.tensor([2.0, 1.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 2
    input = torch.tensor([[-1.0, 0.0, 1.0],
                          [np.inf, -np.inf, np.nan]], dtype=torch.float64).numpy()
    other = torch.tensor(0.5, dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 3
    input = torch.ones((2, 3), dtype=torch.float32).numpy()
    other = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 4
    input = torch.tensor([-1e-300, -0.0], dtype=torch.float64).numpy()
    other = torch.tensor([0.0, 0.0], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 5
    input = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float16).numpy()
    other = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float16).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 6
    input = torch.randn((2, 1, 4), dtype=torch.float32).numpy()
    other = torch.linspace(-1.0, 1.0, 4, dtype=torch.float32).numpy()
    out = np.empty((2, 1, 4), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 7
    input = torch.tensor(1.0, dtype=torch.float64).numpy()
    other = torch.tensor(2.0, dtype=torch.float64).numpy()
    out = np.empty((), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 8
    input = torch.tensor([np.finfo(np.float32).max], dtype=torch.float32).numpy()
    other = torch.tensor([0.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 9
    input = torch.tensor([-np.inf, np.inf, 0.0], dtype=torch.float64).numpy()
    other = torch.tensor([np.inf, -np.inf, np.nan], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 10
    input = torch.linspace(-5.0, 5.0, 5, dtype=torch.float64).numpy()
    other = torch.tensor([0.0], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 11
    input = torch.tensor([[1.0, -1.0],
                          [2.0, -2.0]], dtype=torch.float32).numpy()
    other = torch.tensor([[3.0, -3.0],
                          [4.0, -4.0]], dtype=torch.float32).numpy()
    base_out = np.empty((2, 4), dtype=np.float32)
    out = base_out[:, ::2]
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 12
    input = torch.tensor([[0.0], [1.0]], dtype=torch.float16).numpy()
    other = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16).numpy()
    out = np.empty((2, 3), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.nextafter"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nextafter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nextafter'.")


check_valid('torch.nextafter', generated_inputs['torch.nextafter'], lib="torch", suffix=0)
