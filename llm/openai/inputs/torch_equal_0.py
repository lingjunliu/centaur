
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def equal_inputs():
    list_of_inputs = []

    input = torch.tensor([1, 2]).numpy()
    other = torch.tensor([1, 2]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([3.0, float('nan')]).numpy()
    other = torch.tensor([3.0, float('nan')]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    other = torch.tensor([1, 2, 3], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([]).numpy()
    other = torch.tensor([]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([[1, 2, 3]]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([-1, -2, -3], dtype=torch.int16).numpy()
    other = torch.tensor([-1, -2, -3], dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([True, False, True], dtype=torch.bool).numpy()
    other = torch.tensor([True, False, True], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([1.0, float('inf'), -float('inf')], dtype=torch.float64).numpy()
    other = torch.tensor([1.0, float('inf'), -float('inf')], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).numpy()
    other = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.int64).numpy()
    other = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 0]]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([1+2j, 3-4j], dtype=torch.complex64).numpy()
    other = torch.tensor([1+2j, 3-4j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.zeros((2, 0, 3), dtype=torch.float32).numpy()
    other = torch.zeros((2, 0, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([0, 255], dtype=torch.uint8).numpy()
    other = torch.tensor([0, 255], dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.arange(6, dtype=torch.int64).reshape(2, 3).numpy()
    other = torch.arange(6, dtype=torch.int64).reshape(2, 3).t().contiguous().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([[-1.5, 0.0, 1.5]], dtype=torch.float16).numpy()
    other = torch.tensor([[-1.5, 0.0, 1.5]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    return list_of_inputs

generated_inputs["torch.equal"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.equal'.")


check_valid('torch.equal', generated_inputs['torch.equal'], lib="torch", suffix=0)
