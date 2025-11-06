
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def abs_inputs():
    list_of_inputs = []

    t = torch.tensor([-3, -1, 0, 2, 5], dtype=torch.int32)
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": torch.empty_like(t).numpy()}))

    t = torch.tensor([[-1.0, 2.5, -3.3],
                      [4.1, -5.2, 6.0]], dtype=torch.float32)
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": torch.empty_like(t).numpy()}))

    t = torch.tensor(-7.25, dtype=torch.float64)
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": torch.empty((), dtype=torch.float64).numpy()}))

    t = torch.tensor([[[-1, 2, -3],
                       [4, -5, 6]],
                      [[-7, 8, -9],
                       [10, -11, 12]]], dtype=torch.int64)
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": torch.empty_like(t).numpy()}))

    t = torch.randn(1, 2, 3, 4, dtype=torch.float16) * -5
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": torch.empty_like(t).numpy()}))

    t = torch.tensor([1+2j, -3-4j, 0+0j], dtype=torch.complex64)
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": torch.empty(t.shape, dtype=torch.float32).numpy()}))

    t = torch.tensor([[1-1j, -2+2j],
                      [3+0j, -4-5j]], dtype=torch.complex128)
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": torch.empty(t.shape, dtype=torch.float64).numpy()}))

    t = torch.tensor([float('nan'), float('-inf'), 0.0, float('inf'), -1.5], dtype=torch.float32)
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": torch.empty_like(t).numpy()}))

    t = torch.tensor([-127, -1, 0, 1, 126], dtype=torch.int8)
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": torch.empty_like(t).numpy()}))

    t = torch.empty((0, 5), dtype=torch.float32)
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": torch.empty_like(t).numpy()}))

    t = (torch.arange(12, dtype=torch.int16).reshape(2, 1, 1, 2, 3) - 6)
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": torch.empty_like(t).numpy()}))

    base = torch.arange(12, dtype=torch.float64).reshape(3, 4) - 6.0
    t = base.t()
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": torch.empty_like(t).numpy()}))

    return list_of_inputs

generated_inputs["torch.abs"] = abs_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.abs' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.abs'.")


check_valid('torch.abs', generated_inputs['torch.abs'], lib="torch", suffix=0)
