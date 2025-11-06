
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def masked_select_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, -2.5, 3.0, 0.0], dtype=torch.float32).numpy()
    mask = torch.tensor([True, False, True, False], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask}))

    input = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64).numpy()
    mask = torch.tensor([[True, True, False], [False, True, False]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask}))

    input = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3).numpy()
    mask = torch.tensor([[[True], [False]]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask}))

    input = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]], dtype=torch.uint8).numpy()
    mask = torch.tensor(True, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask}))

    input = torch.tensor([[0.1, -0.2, 0.3, -0.4],
                          [1.5, 2.5, -3.5, 4.5],
                          [-1.1, 2.2, -3.3, 4.4]], dtype=torch.float16).numpy()
    mask = torch.tensor([[True], [False], [True]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask}))

    input = torch.tensor([-10, -5, 0, 5, 10], dtype=torch.int32).numpy()
    mask = torch.tensor([False, True, True, False, True], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask}))

    input = torch.tensor([[1+2j, 3-4j, 0+0j],
                          [-1+0.5j, 2+2j, -3-3j]], dtype=torch.complex64).numpy()
    mask = torch.tensor([[True, False, True],
                         [False, True, True]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask}))

    input = torch.tensor([[True, False],
                          [False, True],
                          [True, True]], dtype=torch.bool).numpy()
    mask = torch.tensor([[False, True],
                         [True, False],
                         [True, True]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask}))

    input = torch.empty(0, dtype=torch.float32).numpy()
    mask = torch.empty(0, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask}))

    input = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).numpy()
    mask = torch.tensor([True, False], dtype=torch.bool).reshape(2, 1, 1).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask}))

    input = torch.arange(20, dtype=torch.int64).reshape(4, 5).numpy()
    mask = torch.tensor([[True, False, True, False, True]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask}))

    input = torch.tensor([float('nan'), float('inf'), -float('inf'), 1.0, -2.0], dtype=torch.float32).numpy()
    mask = torch.tensor([True, True, False, False, True], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask}))

    return list_of_inputs

generated_inputs["torch.masked_select"] = masked_select_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.masked_select' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.masked_select'.")


check_valid('torch.masked_select', generated_inputs['torch.masked_select'], lib="torch", suffix=0)
