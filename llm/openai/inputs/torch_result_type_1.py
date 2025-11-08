
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def result_type_inputs():
    list_of_inputs = []

    tensor1 = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    tensor2 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2}))

    tensor1 = torch.ones((2, 3), dtype=torch.int8).numpy()
    tensor2 = torch.full((2, 3), -5, dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2}))

    tensor1 = torch.tensor(True, dtype=torch.bool).numpy()
    tensor2 = torch.tensor([-1, 0, 1], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2}))

    tensor1 = torch.tensor([[1.5, -2.5]], dtype=torch.float64).numpy()
    tensor2 = torch.tensor(3, dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2}))

    tensor1 = torch.tensor([1+2j, -3+4j], dtype=torch.complex64).numpy()
    tensor2 = torch.tensor([5.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2}))

    tensor1 = torch.tensor([255], dtype=torch.uint8).numpy()
    tensor2 = torch.tensor([-1], dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2}))

    tensor1 = torch.tensor(float('nan'), dtype=torch.float32).numpy()
    tensor2 = torch.tensor(float('inf'), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2}))

    tensor1 = torch.empty((0,), dtype=torch.float16).numpy()
    tensor2 = torch.tensor([1.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2}))

    tensor1 = torch.tensor([True, False, True], dtype=torch.bool).numpy()
    tensor2 = torch.tensor([0.0, -0.0, 1.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2}))

    tensor1 = torch.arange(12, dtype=torch.int64).reshape(3, 4).numpy()
    tensor2 = torch.tensor(1, dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2}))

    tensor1 = torch.tensor([1+0j, 2-3j], dtype=torch.complex128).numpy()
    tensor2 = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2}))

    tensor1 = torch.randn((2, 2, 2), dtype=torch.float32).numpy()
    tensor2 = torch.randint(-10, 10, (2, 1, 2), dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor1": tensor1, "tensor2": tensor2}))

    return list_of_inputs

generated_inputs["torch.result_type_1"] = result_type_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.result_type_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.result_type_1'.")


check_valid('torch.result_type', generated_inputs['torch.result_type_1'], lib="torch", suffix=1)
