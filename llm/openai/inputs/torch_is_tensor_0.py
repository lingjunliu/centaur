
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def is_tensor_inputs():
    list_of_inputs = []

    obj = torch.tensor([1.0, -2.5, 3.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor([[1, -1, 0], [2, -2, 3]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor(42, dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor([[[0, 255, 128], [10, 20, 30]],
                        [[40, 50, 60], [70, 80, 90]]], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor([True, False, True, False], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor([], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.empty((0, 4), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor([1+2j, -3+0.5j, -1j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor([[1-1j, 2+0j], [3.5-2.5j, 4j]], dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.arange(20, dtype=torch.float64)[::2].numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.arange(12, dtype=torch.int64).reshape(3, 4).t().numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor([float('nan'), float('inf'), -float('inf')], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    return list_of_inputs

generated_inputs["torch.is_tensor"] = is_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.is_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_tensor'.")


check_valid('torch.is_tensor', generated_inputs['torch.is_tensor'], lib="torch", suffix=0)
