
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def index_select_inputs():
    list_of_inputs = []

    # 1
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).numpy()
    dim = 0
    index = torch.tensor([0, 2, 4], dtype=torch.long).numpy()
    out = torch.tensor([], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    # 2
    input = torch.arange(12, dtype=torch.long).reshape(3, 4).numpy()
    dim = 1
    index = torch.tensor([3, 1, 1, 0], dtype=torch.long).numpy()
    out_shape = (input.shape[0], index.shape[0])
    out = torch.empty(out_shape, dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    # 3
    input = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4).numpy()
    dim = -1
    index = torch.tensor([0, 3, 2], dtype=torch.long).numpy()
    out = torch.tensor([], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    # 4
    input = (torch.arange(2*0*3) > 0).to(torch.bool).reshape(2, 0, 3).numpy()
    dim = 1
    index = torch.tensor([], dtype=torch.long).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    # 5
    real = torch.randn(2, 3, 4, 5, dtype=torch.float32)
    imag = torch.randn(2, 3, 4, 5, dtype=torch.float32)
    input = (real + 1j * imag).numpy()
    dim = 2
    index = torch.tensor([1, 1, 0, 3], dtype=torch.long).numpy()
    out_shape = (input.shape[0], input.shape[1], index.shape[0], input.shape[3])
    out = (torch.zeros(out_shape, dtype=torch.complex64)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    # 6
    input = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0],
                          [5.0, 6.0],
                          [7.0, 8.0]], dtype=torch.float16).numpy()
    dim = -2
    index = torch.tensor([1, 3, 0], dtype=torch.long).numpy()
    out = torch.tensor([], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    # 7
    input = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int8).numpy()
    dim = 0
    index = torch.tensor([4, 4, 2, 0], dtype=torch.long).numpy()
    out_shape = (index.shape[0],)
    out = torch.zeros(out_shape, dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    # 8
    input = torch.arange(3*2*4, dtype=torch.uint8).reshape(3, 2, 4).numpy()
    dim = 0
    index = torch.tensor([2, 0], dtype=torch.long).numpy()
    out = torch.tensor([], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    # 9
    input = torch.arange(1*2*3*4, dtype=torch.float64).reshape(1, 2, 3, 4).numpy()
    dim = -3
    index = torch.tensor([1, 0, 1, 1, 0], dtype=torch.long).numpy()
    out_shape = (input.shape[0], index.shape[0], input.shape[2], input.shape[3])
    out = torch.empty(out_shape, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    # 10
    input = torch.arange(12, dtype=torch.float32).reshape(2, 6).numpy()
    dim = -1
    index = torch.tensor([5, 2, 2, 0, 3], dtype=torch.long).numpy()
    out = torch.tensor([], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    # 11
    real = torch.randn(2, 2, 2, dtype=torch.float64)
    imag = torch.randn(2, 2, 2, dtype=torch.float64)
    input = (real + 1j * imag).numpy()
    dim = 1
    index = torch.tensor([1, 0, 1], dtype=torch.long).numpy()
    out = torch.tensor([], dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    # 12
    input = torch.tensor([[True, False, True],
                          [False, True, False],
                          [True, True, False]], dtype=torch.bool).numpy()
    dim = 0
    index = torch.tensor([2, 0], dtype=torch.long).numpy()
    out_shape = (index.shape[0], input.shape[1])
    out = torch.empty(out_shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    # 13
    input = torch.arange(2*3*4, dtype=torch.int32).reshape(2, 3, 4).numpy()
    dim = 1
    index = torch.tensor([0, 2], dtype=torch.long).numpy()
    out_shape = (input.shape[0], index.shape[0], input.shape[2])
    out = torch.zeros(out_shape, dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    # 14
    input = torch.arange(2*3*4*5, dtype=torch.float32).reshape(2, 3, 4, 5).numpy()
    dim = 3
    index = torch.tensor([4, 1, 0, 3, 2], dtype=torch.long).numpy()
    out = torch.tensor([], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "index": index, "out": out}))

    return list_of_inputs

generated_inputs["torch.index_select"] = index_select_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.index_select' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.index_select'.")


check_valid('torch.index_select', generated_inputs['torch.index_select'], lib="torch", suffix=0)
