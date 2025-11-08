
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def conj_inputs():
    list_of_inputs = []
    
    input = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[-1.0, 0.0], [2.5, -3.5]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.arange(-12, 12, dtype=torch.int64).reshape(2, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor(2 - 5j, dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.randn((2, 2, 2, 3), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.empty((2, 0, 4), dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[True, False], [False, True]]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[0 + 0j, 1 - 1j, -1 + 2j], [3 + 0j, 0 - 4j, -5 - 6j]], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.arange(24, dtype=torch.float64).reshape(4, 6).t().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.randint(-128, 128, (1, 2, 3, 4, 5), dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    return list_of_inputs

generated_inputs["torch.conj"] = conj_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.conj' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.conj'.")


check_valid('torch.conj', generated_inputs['torch.conj'], lib="torch", suffix=0)
