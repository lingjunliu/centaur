
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def vsplit_inputs():
    list_of_inputs = []

    input = torch.arange(16.0, dtype=torch.float32).reshape(4, 4).numpy()
    indices_or_sections = (2,)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.tensor([[-1.0, -2.0, -3.0],
                          [4.5, 5.5, 6.5],
                          [7.2, 8.8, 9.1],
                          [10.0, -11.0, 12.5],
                          [0.0, -0.1, 0.2],
                          [3.14, 2.71, -1.0]], dtype=torch.float64).numpy()
    indices_or_sections = (1, 4)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(20, dtype=torch.int64).reshape(5, 2, 2).numpy()
    indices_or_sections = (3,)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.tensor([[True, False, True, False],
                          [False, False, True, True],
                          [True, True, True, False]], dtype=torch.bool).numpy()
    indices_or_sections = (0, 1, 3)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    x = torch.randn(4, 1, 2, 2, dtype=torch.float32)
    y = torch.randn(4, 1, 2, 2, dtype=torch.float32)
    input = (x + 1j * y).numpy()
    indices_or_sections = (1, 2, 4)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(24, dtype=torch.float16).reshape(6, 4).t().numpy()
    indices_or_sections = (2, 3)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.empty((0, 5), dtype=torch.float32).numpy()
    indices_or_sections = (0, 2)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(6 * 2 * 3, dtype=torch.int32).reshape(6, 2, 3).numpy()
    indices_or_sections = (-2,)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.randn(5, 5, dtype=torch.float32).numpy()
    indices_or_sections = (2, 2, 4)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(8, dtype=torch.int64).reshape(4, 2).numpy()
    indices_or_sections = (3, 6)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.ones((7, 1, 1, 1, 1), dtype=torch.float32).numpy()
    indices_or_sections = (1, 6)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(24, dtype=torch.uint8).reshape(8, 3).numpy()
    indices_or_sections = (2, 5, 8)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    a = torch.randn(3, 4, dtype=torch.float64)
    b = torch.randn(3, 4, dtype=torch.float64)
    input = (a + 1j * b).numpy()
    indices_or_sections = (1, 2)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = (torch.arange(15, dtype=torch.float32).reshape(5, 3) * -1).numpy()
    indices_or_sections = (1, 3, 5)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    return list_of_inputs

generated_inputs["torch.vsplit_3"] = vsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.vsplit_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vsplit_3'.")


check_valid('torch.vsplit', generated_inputs['torch.vsplit_3'], lib="torch", suffix=3)
