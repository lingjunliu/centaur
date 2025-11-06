
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def less_equal_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    other = torch.tensor([0.5, 2.0, 5.0], dtype=torch.float32).numpy()
    out = np.empty(np.broadcast(input, other).shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[-1, 2], [3, -4]], dtype=torch.int64).numpy()
    other = torch.tensor([[-1, 0], [4, -5]], dtype=torch.int64).numpy()
    out = np.empty(np.broadcast(input, other).shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.arange(-6, 6, dtype=torch.float16).view(2, 2, 3).numpy()
    other = torch.tensor([-5.0, 0.0, 5.0], dtype=torch.float64).numpy()
    out = np.empty(np.broadcast(input, other).shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[-2], [0], [2]], dtype=torch.int32).numpy()
    other = torch.tensor([-3.5, 0.0, 1.0, 100.0], dtype=torch.float32).numpy().reshape(1, 4)
    out = np.empty(np.broadcast(input, other).shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[True, False, True], [False, False, True]], dtype=torch.bool).numpy()
    other = torch.tensor([[True], [False]], dtype=torch.bool).numpy()
    out = np.empty(np.broadcast(input, other).shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([0, 10, 255, 128, 1], dtype=torch.uint8).numpy()
    other = torch.tensor([-1, 10, 254, 200, 2], dtype=torch.int16).numpy()
    out = np.empty(np.broadcast(input, other).shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(3, dtype=torch.int64).numpy()
    other = torch.tensor([[1, 3], [5, -2]], dtype=torch.int64).numpy()
    out = np.empty(np.broadcast(input, other).shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.arange(6, dtype=torch.int8).view(2, 1, 3, 1).numpy()
    other = torch.tensor([-2, 0, 2], dtype=torch.int8).view(1, 1, 3, 1).numpy()
    out = np.empty(np.broadcast(input, other).shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    other = torch.zeros((0, 1), dtype=torch.float32).numpy()
    out = np.empty(np.broadcast(input, other).shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    base = torch.arange(12, dtype=torch.int32).view(3, 4).t()
    input = base.numpy()
    other = torch.tensor([5, 6, 7], dtype=torch.int32).view(1, 3).numpy()
    out = np.empty(np.broadcast(input, other).shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.0], dtype=torch.float64).numpy()
    other = torch.tensor([0.0, float('inf'), -1e20, 0.0], dtype=torch.float64).numpy()
    out = np.empty(np.broadcast(input, other).shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(True, dtype=torch.bool).numpy()
    other = torch.tensor([[[True, False, True]] , [[False, True, False]]], dtype=torch.bool).numpy()
    out = np.empty(np.broadcast(input, other).shape, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.less_equal_1"] = less_equal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.less_equal_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.less_equal_1'.")


check_valid('torch.less_equal', generated_inputs['torch.less_equal_1'], lib="torch", suffix=1)
