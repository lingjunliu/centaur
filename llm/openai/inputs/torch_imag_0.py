
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def imag_inputs():
    list_of_inputs = []

    # 1: 1D, complex64, random
    input = torch.randn(5, dtype=torch.cfloat).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2: 1D, complex128, explicit values
    input = torch.tensor([1+2j, -3-4j, 0+0j, 5-6j], dtype=torch.cdouble).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3: 0-D scalar, complex64
    input = torch.tensor(1-3j, dtype=torch.cfloat).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4: 2D matrix, complex64, random
    input = torch.randn(2, 3, dtype=torch.cfloat).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5: 3D tensor, complex128, random
    input = torch.randn(2, 2, 2, dtype=torch.cdouble).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6: Empty 1D, complex64
    input = torch.empty(0, dtype=torch.cfloat).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7: Non-contiguous (transpose), complex64
    input = torch.randn(2, 3, dtype=torch.cfloat).t().numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8: Contains inf and nan, complex128
    input = torch.tensor([complex(float('inf'), float('-inf')),
                          complex(float('nan'), float('nan'))], dtype=torch.cdouble).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9: 4D tensor, complex128, random
    input = torch.randn(2, 1, 3, 4, dtype=torch.cdouble).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10: Strided slice, complex64
    input = torch.randn(9, dtype=torch.cfloat)[::2].numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11: Purely imaginary values, complex64, 2D column
    input = torch.tensor([[0+1j], [0-2j], [0+3j]], dtype=torch.cfloat).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12: Empty multidimensional, complex64
    input = torch.empty(2, 0, 3, dtype=torch.cfloat).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.imag"] = imag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.imag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.imag'.")


check_valid('torch.imag', generated_inputs['torch.imag'], lib="torch", suffix=0)
