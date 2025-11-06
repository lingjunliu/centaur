
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def isneginf_inputs():
    list_of_inputs = []

    t = torch.tensor([-float('inf'), float('inf'), 0.0, -1.5, float('nan')], dtype=torch.float64)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t, dtype=torch.bool).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([[0.0, -float('inf'), 2.3],
                      [float('inf'), -5.6, -float('inf')]], dtype=torch.float32)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t, dtype=torch.bool).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor(-float('inf'), dtype=torch.float32)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t, dtype=torch.bool).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([[[1.0, -2.0, -float('inf')],
                       [3.0, 4.0, 5.0]],
                      [[-float('inf'), 0.0, 7.0],
                       [8.0, 9.0, float('inf')]]], dtype=torch.float16)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t, dtype=torch.bool).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([-3, 0, 7, -10], dtype=torch.int32)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t, dtype=torch.bool).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([-0.0, 0.0, -float('inf'), 5.0], dtype=torch.float64)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t, dtype=torch.bool).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.empty((0,), dtype=torch.float64)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t, dtype=torch.bool).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.empty((2, 0, 4), dtype=torch.float32)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t, dtype=torch.bool).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = torch.tensor([[0.0, -float('inf'), 2.0, 3.0],
                         [-float('inf'), 5.0, float('inf'), -6.0],
                         [7.0, 8.0, -9.0, -float('inf')]], dtype=torch.float32)
    t = base[:, ::2]
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t, dtype=torch.bool).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([[[[-float('inf'), 1.0, 2.0]] ,
                       [[3.0, -4.0, -float('inf')]]]], dtype=torch.float32)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t, dtype=torch.bool).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t, dtype=torch.bool).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([-float('inf'), -32768.0, 65504.0, float('inf')], dtype=torch.float16)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t, dtype=torch.bool).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.isneginf"] = isneginf_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.isneginf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isneginf'.")


check_valid('torch.isneginf', generated_inputs['torch.isneginf'], lib="torch", suffix=0)
