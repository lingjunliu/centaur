
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def flatten_inputs():
    list_of_inputs = []

    input = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    start_dim = 1
    end_dim = -1
    list_of_inputs.append(copy.deepcopy({"start_dim": start_dim, "end_dim": end_dim, "input": input}))

    input = torch.arange(24, dtype=torch.int64).reshape(2, 3, 4).numpy()
    start_dim = 0
    end_dim = -1
    list_of_inputs.append(copy.deepcopy({"start_dim": start_dim, "end_dim": end_dim, "input": input}))

    input = torch.ones(2, 3, 4, 5, 6, dtype=torch.float64).numpy()
    start_dim = 2
    end_dim = 3
    list_of_inputs.append(copy.deepcopy({"start_dim": start_dim, "end_dim": end_dim, "input": input}))

    real = torch.randn(2, 3, 4, 5, 6, dtype=torch.float32)
    imag = torch.randn(2, 3, 4, 5, 6, dtype=torch.float32)
    input = (real + 1j * imag).numpy()
    start_dim = -3
    end_dim = -2
    list_of_inputs.append(copy.deepcopy({"start_dim": start_dim, "end_dim": end_dim, "input": input}))

    input = torch.tensor([True, False, True, False, True], dtype=torch.bool).numpy()
    start_dim = 0
    end_dim = 0
    list_of_inputs.append(copy.deepcopy({"start_dim": start_dim, "end_dim": end_dim, "input": input}))

    input = torch.randint(-100, 100, (2, 3, 4, 5), dtype=torch.int32).numpy()
    start_dim = 3
    end_dim = 3
    list_of_inputs.append(copy.deepcopy({"start_dim": start_dim, "end_dim": end_dim, "input": input}))

    input = torch.randint(0, 256, (4, 5, 6), dtype=torch.uint8).numpy()
    start_dim = 1
    end_dim = 2
    list_of_inputs.append(copy.deepcopy({"start_dim": start_dim, "end_dim": end_dim, "input": input}))

    input = torch.randn(2, 3, 4, 5, 6, dtype=torch.float16).numpy()
    start_dim = 1
    end_dim = -2
    list_of_inputs.append(copy.deepcopy({"start_dim": start_dim, "end_dim": end_dim, "input": input}))

    input = torch.randint(-10, 10, (3, 4, 5, 6), dtype=torch.int8).numpy()
    start_dim = -4
    end_dim = -1
    list_of_inputs.append(copy.deepcopy({"start_dim": start_dim, "end_dim": end_dim, "input": input}))

    input = torch.empty(2, 0, 3, dtype=torch.float32).numpy()
    start_dim = 1
    end_dim = -1
    list_of_inputs.append(copy.deepcopy({"start_dim": start_dim, "end_dim": end_dim, "input": input}))

    input = torch.randn(2, 1, 3, 1, 4, 5, dtype=torch.float32).numpy()
    start_dim = 1
    end_dim = 4
    list_of_inputs.append(copy.deepcopy({"start_dim": start_dim, "end_dim": end_dim, "input": input}))

    input = torch.randn(8, 7, dtype=torch.float64).numpy()
    start_dim = 0
    end_dim = 1
    list_of_inputs.append(copy.deepcopy({"start_dim": start_dim, "end_dim": end_dim, "input": input}))

    return list_of_inputs

generated_inputs["torch.nn.Flatten"] = flatten_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Flatten' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Flatten'.")


check_valid('torch.nn.Flatten', generated_inputs['torch.nn.Flatten'], lib="torch", suffix=0)
