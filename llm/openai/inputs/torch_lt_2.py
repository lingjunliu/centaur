
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def lt_inputs_2():
    list_of_inputs = []

    input = torch.tensor([1, -2, 3, -4], dtype=torch.int32).numpy()
    other = 0.0
    out = np.empty(input.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[1.5, -0.5, 0.0], [2.3, -3.7, 4.2]], dtype=torch.float32).numpy()
    other = -0.5
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(3.14, dtype=torch.float64).numpy()
    other = 3.14
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.arange(-12, 12, dtype=torch.int64).reshape(2, 3, 4).numpy()
    other = -5.0
    out = np.zeros((2, 3, 4), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.empty((0,), dtype=torch.float32).numpy()
    other = 1.0
    out = np.empty((0,), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.empty((2, 0, 3), dtype=torch.float16).numpy()
    other = 3.14
    out = np.empty((2, 0, 3), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[True, False], [False, True]], dtype=torch.bool).numpy()
    other = 1.0
    out = np.empty((2, 2), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    base = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    input = base.t().numpy()
    other = 5.0
    out = np.empty((4, 3), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([float('nan'), float('inf'), -float('inf'), 0.0], dtype=torch.float64).numpy()
    other = float('nan')
    out = np.empty((4,), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.arange(24, dtype=torch.int16).reshape(1, 2, 3, 4).numpy()
    other = 10.0
    out = np.empty((1, 2, 3, 4), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([2**60, -2**60, 0], dtype=torch.int64).numpy()
    other = 1e9
    out = np.empty(input.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]], dtype=torch.float16).numpy()
    other = -2.5
    out = np.empty((2, 3), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.lt_2"] = lt_inputs_2()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.lt_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lt_2'.")


check_valid('torch.lt', generated_inputs['torch.lt_2'], lib="torch", suffix=2)
