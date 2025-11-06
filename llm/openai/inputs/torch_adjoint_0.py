
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def adjoint_inputs():
    list_of_inputs = []

    # 1: 2x2 float32
    input = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 2: 3x2 int64 with negatives
    input = torch.tensor([[-1, 2],
                          [3, -4],
                          [5, 6]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 3: 2x3 float16
    input = torch.tensor([[0.5, -1.5, 2.0],
                          [3.25, -4.5, 5.75]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 4: 2x2 complex64
    x = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    input = torch.complex(x, -x).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 5: 3D batch (4, 2, 3) float32
    input = torch.randn(4, 2, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 6: 4D int32 (2, 3, 4, 5)
    input = torch.randint(-10, 10, (2, 3, 4, 5), dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 7: 2D bool (2, 5)
    input = torch.tensor([[True, False, True, False, True],
                          [False, False, True, True, False]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 8: Zero-sized last dim (3, 0) float32
    input = torch.empty((3, 0), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 9: Batched complex128 (2, 3, 3)
    real = torch.randn(2, 3, 3, dtype=torch.float64)
    imag = torch.randn(2, 3, 3, dtype=torch.float64)
    input = torch.complex(real, imag).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 10: Non-contiguous (created via transpose) then to numpy
    base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    input = base.transpose(1, 2).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 11: High-dimensional with singleton dims (1, 1, 2, 2) float64
    input = torch.tensor([[[[1.0, -2.0],
                            [3.0, -4.0]]]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 12: Empty first dim (0, 5) int64
    input = torch.empty((0, 5), dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.adjoint"] = adjoint_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.adjoint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.adjoint'.")


check_valid('torch.adjoint', generated_inputs['torch.adjoint'], lib="torch", suffix=0)
