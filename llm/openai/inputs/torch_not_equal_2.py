
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def not_equal_2_inputs():
    list_of_inputs = []

    # 1
    input = torch.tensor([-1.0, 0.0, 1.5, 2.0], dtype=torch.float32).numpy()
    other = 0.0
    out = np.empty(input.shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 2
    input = torch.tensor([[1, 5, 7], [5, 5, -3]], dtype=torch.int64).numpy()
    other = 5.0
    out = np.empty(input.shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 3
    input = torch.tensor([[[-2.0, -1.5, 0.0]], [[1.25, 2.5, 3.75]]], dtype=torch.float64).numpy()
    other = -1.5
    out = np.empty(input.shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 4
    input = torch.tensor([True, False, True, False], dtype=torch.bool).numpy()
    other = 1.0
    out = np.empty(input.shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 5
    input = torch.tensor([[1+2j, 0+0j], [-3.5+0.5j, 4-1j]], dtype=torch.complex64).numpy()
    other = 0.0
    out = np.empty(input.shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 6
    input = torch.empty((0,), dtype=torch.float32).numpy()
    other = 3.14
    out = np.empty(input.shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 7
    input = torch.empty((2, 0, 4), dtype=torch.int32).numpy()
    other = -2.0
    out = np.empty(input.shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 8
    base = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    input = base.t().numpy()
    other = 2.0
    out = np.empty(input.shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 9
    input = torch.tensor([[[[0.0], [0.5], [1.0]], [[1.5], [2.0], [2.5]]]], dtype=torch.float16).numpy()
    other = 0.5
    out = np.empty(input.shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 10
    input = torch.tensor([[[0, 128, 255], [64, 128, 192]],
                          [[10, 20, 30], [40, 50, 60]]], dtype=torch.uint8).numpy()
    other = 128.0
    out = np.empty(input.shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 11
    input = torch.tensor([float('nan'), float('inf'), -float('inf'), 0.0], dtype=torch.float64).numpy()
    other = float('nan')
    out = np.empty(input.shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 12
    input_full = torch.arange(-10, 10, dtype=torch.int16).numpy()
    input = input_full[::2]
    other = -3.0
    out = np.empty(input.shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.not_equal_2"] = not_equal_2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.not_equal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.not_equal_2'.")


check_valid('torch.not_equal', generated_inputs['torch.not_equal_2'], lib="torch", suffix=2)
