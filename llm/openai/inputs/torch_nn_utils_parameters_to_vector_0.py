
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def parameters_to_vector_inputs():
    list_of_inputs = []

    p1 = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"parameters": p1}))

    p2 = torch.arange(6, dtype=torch.float64).reshape(2, 3).numpy()
    list_of_inputs.append(copy.deepcopy({"parameters": p2}))

    p3 = torch.tensor([[[1, -1], [2, -2]], [[3, -3], [4, -4]]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"parameters": p3}))

    p4 = torch.tensor([[True, False, True, False], [False, True, False, True]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"parameters": p4}))

    p5 = torch.randn(3, 2, 2, 2, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"parameters": p5}))

    p6 = torch.tensor([[float('nan'), float('inf'), -float('inf'), 0.0]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"parameters": p6}))

    p7 = torch.randn(1, 1, 2, 3, 1, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"parameters": p7}))

    p8 = torch.tensor([[2147483647], [-2147483648], [123456789], [-987654321]], dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"parameters": p8}))

    real = torch.tensor([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]], dtype=torch.float32)
    imag = torch.tensor([[0.1, -0.2], [0.0, 1.5], [0.0, -3.14]], dtype=torch.float32)
    p9 = torch.complex(real, imag).numpy()
    list_of_inputs.append(copy.deepcopy({"parameters": p9}))

    p10 = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"parameters": p10}))

    return list_of_inputs

generated_inputs["torch.nn.utils.parameters_to_vector"] = parameters_to_vector_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.parameters_to_vector' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.parameters_to_vector'.")


check_valid('torch.nn.utils.parameters_to_vector', generated_inputs['torch.nn.utils.parameters_to_vector'], lib="torch", suffix=0)
