
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def hsplit_inputs():
    list_of_inputs = []

    inp = torch.arange(10, dtype=torch.float32).numpy()
    indices = [2, 5, 9]
    list_of_inputs.append(copy.deepcopy({"input": inp, "indices_or_sections": indices}))

    inp = torch.tensor([-5, -4, -3, -2, -1], dtype=torch.int64).numpy()
    indices = [0, 3, 5]
    list_of_inputs.append(copy.deepcopy({"input": inp, "indices_or_sections": indices}))

    inp = torch.arange(16.0, dtype=torch.float64).reshape(4, 4).numpy()
    indices = [1, 3, 6]
    list_of_inputs.append(copy.deepcopy({"input": inp, "indices_or_sections": indices}))

    inp = torch.randint(0, 2, (3, 7)).bool().numpy()
    indices = [2, 2, 5]
    list_of_inputs.append(copy.deepcopy({"input": inp, "indices_or_sections": indices}))

    inp = torch.randn(2, 6, 3, dtype=torch.float16).numpy()
    indices = [1, 4]
    list_of_inputs.append(copy.deepcopy({"input": inp, "indices_or_sections": indices}))

    inp = torch.arange(-30, -30 + 3 * 5 * 2, dtype=torch.int32).reshape(3, 5, 2).numpy()
    indices = [-2]
    list_of_inputs.append(copy.deepcopy({"input": inp, "indices_or_sections": indices}))

    real = torch.randn(2, 8, 1, 1, dtype=torch.float32)
    imag = torch.randn(2, 8, 1, 1, dtype=torch.float32)
    inp = (real + 1j * imag).numpy()
    indices = [3, 5, 8]
    list_of_inputs.append(copy.deepcopy({"input": inp, "indices_or_sections": indices}))

    inp = torch.empty((3, 0), dtype=torch.float32).numpy()
    indices = [0, 2]
    list_of_inputs.append(copy.deepcopy({"input": inp, "indices_or_sections": indices}))

    inp = torch.empty((0,), dtype=torch.float32).numpy()
    indices = [0]
    list_of_inputs.append(copy.deepcopy({"input": inp, "indices_or_sections": indices}))

    inp = torch.randint(0, 256, (1, 9, 2, 2, 2), dtype=torch.uint8).numpy()
    indices = [2, 7]
    list_of_inputs.append(copy.deepcopy({"input": inp, "indices_or_sections": indices}))

    inp = torch.tensor([[42.0]], dtype=torch.float32).numpy()
    indices = [0, 1, 2]
    list_of_inputs.append(copy.deepcopy({"input": inp, "indices_or_sections": indices}))

    inp = torch.arange(6, dtype=torch.int8).reshape(2, 3).numpy()
    indices = [1]
    list_of_inputs.append(copy.deepcopy({"input": inp, "indices_or_sections": indices}))

    return list_of_inputs

generated_inputs["torch.hsplit_2"] = hsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.hsplit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hsplit_2'.")


check_valid('torch.hsplit', generated_inputs['torch.hsplit_2'], lib="torch", suffix=2)
