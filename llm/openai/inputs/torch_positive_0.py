
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def positive_inputs():
    list_of_inputs = []

    input = torch.tensor(3.14, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([-1.0, 0.0, 2.5], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[-3, 0, 7],
                          [8, -1, -9]], dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(2*3*4, dtype=torch.uint8).reshape(2, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(2, 1, 3, 4, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty(2, 0, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([-2**40, 0, 2**40], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (torch.randn(2, 2, dtype=torch.float32) + 1j * torch.randn(2, 2, dtype=torch.float32)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (torch.randn(3, dtype=torch.float64) + 1j * torch.randn(3, dtype=torch.float64)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(60, dtype=torch.int64).reshape(3, 4, 5)[:, :, ::2].numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(1, 2, 1, 3, 2, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([float('nan'), float('inf'), -float('inf'), 0.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.positive"] = positive_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.positive' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.positive'.")


check_valid('torch.positive', generated_inputs['torch.positive'], lib="torch", suffix=0)
