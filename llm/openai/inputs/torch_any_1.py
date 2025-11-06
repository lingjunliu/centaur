
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def torch_any_inputs():
    list_of_inputs = []

    inp = torch.tensor([[False, True]], dtype=torch.bool).numpy()
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.arange(0, 3, dtype=torch.int64).numpy()
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([[0.0, -0.0], [1.5, -2.3]], dtype=torch.float32).numpy()
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([0, 0, 0, 0], dtype=torch.int32).numpy()
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([0, 1, 0, 2], dtype=torch.uint8).numpy()
    out = torch.empty((), dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.randn(2, 0, 3, dtype=torch.float64).numpy()
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([float('nan'), 0.0, -0.0], dtype=torch.float64).numpy()
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([0+0j, 1+0j, 0-2j], dtype=torch.complex64).numpy()
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.ones((4, 3, 2), dtype=torch.bool).numpy()
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.randn(6, dtype=torch.float16).numpy()
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([[[[[-1, 0, 2, -3],
                           [4, 0, 0, -5],
                           [6, 7, 0, 0]]]]], dtype=torch.int8).repeat(2,1,1,1,1).numpy()
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.zeros((10,), dtype=torch.uint8).numpy()
    out = torch.empty((), dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    return list_of_inputs

generated_inputs["torch.any_1"] = torch_any_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.any_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.any_1'.")


check_valid('torch.any', generated_inputs['torch.any_1'], lib="torch", suffix=1)
