
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def isinf_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, float('inf'), 2.0, float('-inf'), float('nan')], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0.0, 1.0, float('inf')], [float('-inf'), 5.0, 6.0]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[1.0, 2.0, 3.0], [float('inf'), -1.0, 0.0]], [[-2.0, float('-inf'), 4.0], [5.0, 6.0, float('nan')]]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(float('inf'), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1, -1, 0, 123456], dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([complex(float('inf'), 0.0), complex(1.0, float('inf')), complex(3.0, 4.0)], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[complex(float('-inf'), 5.0), complex(6.0, float('-inf'))]], dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1e38, 1e39, -1e39, -1e38], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = torch.arange(20.0, dtype=torch.float64).numpy()
    base[::5] = np.inf
    input = base[::3]
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.full((2, 1, 3, 2), float('-inf'), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((1, 0, 2), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[True, False], [False, True]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.isinf"] = isinf_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.isinf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isinf'.")


check_valid('torch.isinf', generated_inputs['torch.isinf'], lib="torch", suffix=0)
