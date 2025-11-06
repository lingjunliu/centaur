
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def nan_to_num_inputs():
    list_of_inputs = []

    t = torch.tensor([float('nan'), float('inf'), -float('inf'), -3.0, 0.0, 2.5], dtype=torch.float32)
    out_t = torch.empty_like(t)
    list_of_inputs.append(copy.deepcopy({
        "input": t.numpy(),
        "nan": 0.0,
        "posinf": 1.0,
        "neginf": -1.0,
        "out": out_t.numpy()
    }))

    t = torch.tensor([[1.0, float('nan'), -2.0],
                      [float('inf'), -float('inf'), 3.5]], dtype=torch.float64)
    out_t = torch.empty_like(t)
    list_of_inputs.append(copy.deepcopy({
        "input": t.numpy(),
        "nan": -5.5,
        "posinf": 5.5,
        "neginf": -5.5,
        "out": out_t.numpy()
    }))

    t = torch.tensor([[[float('nan'), 1.0], [float('inf'), -float('inf')]],
                      [[-1.0, 2.0], [3.0, -4.0]]], dtype=torch.float16)
    out_t = torch.empty_like(t)
    list_of_inputs.append(copy.deepcopy({
        "input": t.numpy(),
        "nan": 0.0,
        "posinf": 60000.0,
        "neginf": -60000.0,
        "out": out_t.numpy()
    }))

    t = torch.tensor(float('nan'), dtype=torch.float32)
    out_t = torch.empty_like(t)
    list_of_inputs.append(copy.deepcopy({
        "input": t.numpy(),
        "nan": 3.14,
        "posinf": 2.71,
        "neginf": -2.71,
        "out": out_t.numpy()
    }))

    t_base = torch.tensor([[1.0, float('inf'), -1.0],
                           [float('nan'), -float('inf'), 2.0]], dtype=torch.float32)
    t = t_base.t()
    out_t = torch.empty_like(t)
    list_of_inputs.append(copy.deepcopy({
        "input": t.numpy(),
        "nan": -0.25,
        "posinf": 100.0,
        "neginf": -100.0,
        "out": out_t.numpy()
    }))

    t = torch.linspace(-1000.0, 1000.0, steps=10, dtype=torch.float32)
    out_t = torch.empty_like(t)
    list_of_inputs.append(copy.deepcopy({
        "input": t.numpy(),
        "nan": 0.0,
        "posinf": 1000.0,
        "neginf": -1000.0,
        "out": out_t.numpy()
    }))

    t = torch.empty((0, 5), dtype=torch.float32)
    out_t = torch.empty_like(t)
    list_of_inputs.append(copy.deepcopy({
        "input": t.numpy(),
        "nan": 1.0,
        "posinf": 2.0,
        "neginf": -2.0,
        "out": out_t.numpy()
    }))

    t = torch.tensor([0.0, float('inf'), 5.0, float('inf')], dtype=torch.float64)
    out_t = torch.empty_like(t)
    list_of_inputs.append(copy.deepcopy({
        "input": t.numpy(),
        "nan": 7.7,
        "posinf": 9.9,
        "neginf": -9.9,
        "out": out_t.numpy()
    }))

    t = torch.tensor(
        [[[[float('nan'), 1.0, 2.0],
           [3.0, float('inf'), 5.0],
           [6.0, 7.0, -float('inf')]]],
         [[[8.0, 9.0, 10.0],
           [float('nan'), 12.0, 13.0],
           [14.0, 15.0, 16.0]]]], dtype=torch.float64)
    out_t = torch.empty_like(t)
    list_of_inputs.append(copy.deepcopy({
        "input": t.numpy(),
        "nan": -1.0,
        "posinf": 1e6,
        "neginf": -1e6,
        "out": out_t.numpy()
    }))

    t = torch.tensor([1.5, -2.5, 3.0], dtype=torch.float16)
    out_t = torch.empty_like(t)
    list_of_inputs.append(copy.deepcopy({
        "input": t.numpy(),
        "nan": 0.0,
        "posinf": 123.0,
        "neginf": -123.0,
        "out": out_t.numpy()
    }))

    t_full = torch.tensor([float('nan'), 1.0, float('inf'), 2.0,
                           -float('inf'), 3.0, float('nan'), 4.0], dtype=torch.float32)
    t = t_full[::2]
    out_t = torch.empty_like(t)
    list_of_inputs.append(copy.deepcopy({
        "input": t.numpy(),
        "nan": -9.0,
        "posinf": 9.0,
        "neginf": -9.0,
        "out": out_t.numpy()
    }))

    t = torch.tensor([[-float('inf'), float('nan'), float('inf')],
                      [float('nan'), -5.0, 5.0]], dtype=torch.float32)
    out_t = torch.empty_like(t)
    list_of_inputs.append(copy.deepcopy({
        "input": t.numpy(),
        "nan": 0.5,
        "posinf": 42.0,
        "neginf": -42.0,
        "out": out_t.numpy()
    }))

    return list_of_inputs

generated_inputs["torch.nan_to_num"] = nan_to_num_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nan_to_num' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nan_to_num'.")


check_valid('torch.nan_to_num', generated_inputs['torch.nan_to_num'], lib="torch", suffix=0)
