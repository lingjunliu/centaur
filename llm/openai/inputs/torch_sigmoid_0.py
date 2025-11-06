
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def sigmoid_inputs():
    list_of_inputs = []

    input = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([[-10.0, 0.0, 10.0],
                          [2.5, -2.5, 5.5]], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.arange(-6, 6, dtype=torch.float16).reshape(3, 4).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor(-2.5, dtype=torch.float32).numpy()
    out = np.empty((), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4).numpy()
    out = np.empty_like(input, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([True, False, True, True], dtype=torch.bool).numpy()
    out = np.empty_like(input, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.empty((0,), dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([1000.0, -1000.0, 20.0, -20.0], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([float('nan'), float('inf'), float('-inf'), 0.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([[0, 255],
                          [128, 64]], dtype=torch.uint8).numpy()
    out = np.empty_like(input, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    input = t.numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.sigmoid"] = sigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sigmoid'.")


check_valid('torch.sigmoid', generated_inputs['torch.sigmoid'], lib="torch", suffix=0)
