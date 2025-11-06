
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def lu_unpack_inputs():
    list_of_inputs = []

    LU_data = torch.tensor([[2.0, -1.0, 0.5],
                            [3.0, 4.0, -2.0],
                            [-5.0, 1.0, 3.0]], dtype=torch.float32).numpy()
    LU_pivots = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    input_dict = {"LU_data": LU_data, "LU_pivots": LU_pivots, "unpack_data": True, "unpack_pivots": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    LU_data = torch.tensor([[1.0, -2.0],
                            [0.0, 3.5],
                            [4.2, -1.1],
                            [-0.3, 5.7]], dtype=torch.float64).numpy()
    LU_pivots = torch.tensor([2, 4], dtype=torch.int32).numpy()
    input_dict = {"LU_data": LU_data, "LU_pivots": LU_pivots, "unpack_data": True, "unpack_pivots": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    LU_data = torch.tensor([[3.0, -1.0, 2.0, 0.0, -4.5],
                            [6.0, 7.0, -3.2, 1.1, 2.2]], dtype=torch.float32).numpy()
    LU_pivots = torch.tensor([2, 2], dtype=torch.int32).numpy()
    input_dict = {"LU_data": LU_data, "LU_pivots": LU_pivots, "unpack_data": False, "unpack_pivots": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    LU_data = torch.arange(2*3*3, dtype=torch.float64).reshape(2, 3, 3).sub(5.0).numpy()
    LU_pivots = torch.tensor([[1, 3, 2],
                              [3, 2, 2]], dtype=torch.int32).numpy()
    input_dict = {"LU_data": LU_data, "LU_pivots": LU_pivots, "unpack_data": True, "unpack_pivots": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    LU_data = torch.arange(3*4*2, dtype=torch.float32).reshape(3, 4, 2).add(-3.0).numpy()
    LU_pivots = torch.tensor([[1, 4],
                              [2, 3],
                              [3, 1]], dtype=torch.int32).numpy()
    input_dict = {"LU_data": LU_data, "LU_pivots": LU_pivots, "unpack_data": False, "unpack_pivots": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    LU_data = torch.tensor([[-7.0]], dtype=torch.float64).numpy()
    LU_pivots = torch.tensor([1], dtype=torch.int32).numpy()
    input_dict = {"LU_data": LU_data, "LU_pivots": LU_pivots, "unpack_data": True, "unpack_pivots": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = torch.tensor([[1.0, -2.0],
                         [3.5, 0.0],
                         [-1.2, 4.4]], dtype=torch.float32)
    imag = torch.tensor([[0.5, 1.0],
                         [-2.5, 3.3],
                         [0.0, -1.1]], dtype=torch.float32)
    LU_data = (real + 1j * imag).numpy()
    LU_pivots = torch.tensor([2, 3], dtype=torch.int32).numpy()
    input_dict = {"LU_data": LU_data, "LU_pivots": LU_pivots, "unpack_data": True, "unpack_pivots": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    LU_data = torch.arange(2*1*4*4, dtype=torch.float32).reshape(2, 1, 4, 4).add(-10.0).numpy()
    LU_pivots = torch.tensor([[[1, 4, 3, 2]],
                              [[2, 2, 4, 1]]], dtype=torch.int32).numpy()
    input_dict = {"LU_data": LU_data, "LU_pivots": LU_pivots, "unpack_data": False, "unpack_pivots": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    LU_data = torch.tensor([[1.0, -2.0, 3.0, 4.0, -5.0],
                            [6.0, 7.0, -8.0, 9.0, -1.0],
                            [2.0, -3.0, 4.0, -6.0, 7.0],
                            [8.0, -9.0, 1.5, 2.5, -3.5],
                            [0.0, 1.0, -1.0, 3.0, -2.0]], dtype=torch.float64).numpy()
    LU_pivots = torch.tensor([5, 4, 3, 2, 1], dtype=torch.int32).numpy()
    input_dict = {"LU_data": LU_data, "LU_pivots": LU_pivots, "unpack_data": False, "unpack_pivots": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    LU_data = torch.tensor([[2.0, 0.0, -1.0, 4.0, 5.0, -6.0],
                            [7.0, -8.0, 9.0, -1.0, 2.0, 3.0],
                            [-4.0, 5.0, 6.0, -7.0, 8.0, 0.0]], dtype=torch.float64).numpy()
    LU_pivots = torch.tensor([1, 3, 2], dtype=torch.int32).numpy()
    input_dict = {"LU_data": LU_data, "LU_pivots": LU_pivots, "unpack_data": True, "unpack_pivots": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    LU_data = torch.tensor([[1.0, -2.0, 3.0],
                            [4.0, 5.0, -6.0],
                            [-7.0, 8.0, 9.0],
                            [0.5, -1.5, 2.5],
                            [3.5, -4.5, 5.5],
                            [-6.5, 7.5, -8.5]], dtype=torch.float32).numpy()
    LU_pivots = torch.tensor([6, 5, 3], dtype=torch.int32).numpy()
    input_dict = {"LU_data": LU_data, "LU_pivots": LU_pivots, "unpack_data": True, "unpack_pivots": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = torch.tensor([[[1.0, -2.0],
                          [3.0, 4.0]],
                         [[-1.5, 2.5],
                          [0.0, -3.0]]], dtype=torch.float64)
    imag = torch.tensor([[[0.0, 1.0],
                          [-1.0, 2.0]],
                         [[2.0, -2.0],
                          [1.5, 0.5]]], dtype=torch.float64)
    LU_data = (real + 1j * imag).numpy()
    LU_pivots = torch.tensor([[1, 2],
                              [2, 1]], dtype=torch.int32).numpy()
    input_dict = {"LU_data": LU_data, "LU_pivots": LU_pivots, "unpack_data": False, "unpack_pivots": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.lu_unpack"] = lu_unpack_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.lu_unpack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lu_unpack'.")


check_valid('torch.lu_unpack', generated_inputs['torch.lu_unpack'], lib="torch", suffix=0)
