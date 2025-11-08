
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def min_inputs():
    list_of_inputs = []

    # 1
    input = np.array([[3.0, -1.0, 2.5, 0.0],
                      [5.5, 4.2, -7.3, 8.1],
                      [-2.2, -2.2, 1.1, 3.3]], dtype=np.float32)
    dim = 1
    keepdim = False
    values_out = np.empty((3,), dtype=np.float32)
    indices_out = np.empty((3,), dtype=np.int64)
    out = (values_out, indices_out)
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    # 2
    input = np.array([[-10.0, 2.0],
                      [3.0, -4.0]], dtype=np.float64)
    dim = 0
    keepdim = False
    values_out = np.empty((2,), dtype=np.float64)
    indices_out = np.empty((2,), dtype=np.int64)
    out = (values_out, indices_out)
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    # 3
    input = np.array([[[1, 5, -3, 7],
                       [9, -1, -1, 2],
                       [6, 6, 6, 6]],
                      [[-4, -5, -6, -7],
                       [10, 11, 12, 13],
                       [0, -2, -3, -4]]], dtype=np.int64)
    dim = 2
    keepdim = True
    values_out = np.empty((2, 3, 1), dtype=np.int64)
    indices_out = np.empty((2, 3, 1), dtype=np.int64)
    out = (values_out, indices_out)
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    # 4
    input = np.array([5, -1, 3, -7, 2], dtype=np.int32)
    dim = 0
    keepdim = False
    values_out = np.empty((), dtype=np.int32)
    indices_out = np.empty((), dtype=np.int64)
    out = (values_out, indices_out)
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    # 5
    input = np.array([[[[1.5, -0.5, 3.0, -2.0],
                        [0.0, 0.1, -0.1, 2.2],
                        [9.9, -9.9, 4.4, 4.4]]]], dtype=np.float16).repeat(2, axis=0)
    dim = -1
    keepdim = True
    values_out = np.empty((2, 1, 3, 1), dtype=np.float16)
    indices_out = np.empty((2, 1, 3, 1), dtype=np.int64)
    out = (values_out, indices_out)
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    # 6
    input = np.array([[[True, False, True],
                       [False, False, True]],
                      [[True, True, True],
                       [False, True, False]]], dtype=bool)
    dim = 1
    keepdim = False
    values_out = np.empty((2, 3), dtype=bool)
    indices_out = np.empty((2, 3), dtype=np.int64)
    out = (values_out, indices_out)
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    # 7
    input = np.array([[[[[1.0, -1.0],
                         [2.0, -2.0],
                         [3.0, -3.0]]],
                       [[[4.0, 5.0],
                         [6.0, 7.0],
                         [8.0, 9.0]]]]], dtype=np.float32)
    dim = 2
    keepdim = False
    values_out = np.empty((1, 2, 3, 2), dtype=np.float32)
    indices_out = np.empty((1, 2, 3, 2), dtype=np.int64)
    out = (values_out, indices_out)
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    # 8
    input = np.linspace(-5.0, 14.0, num=20, dtype=np.float64).reshape(4, 5)
    dim = -2
    keepdim = True
    values_out = np.empty((1, 5), dtype=np.float64)
    indices_out = np.empty((1, 5), dtype=np.int64)
    out = (values_out, indices_out)
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    # 9
    input = np.array([[0, 0, 1],
                      [-1, -1, 2],
                      [5, 5, 5]], dtype=np.int64)
    dim = 1
    keepdim = True
    values_out = np.empty((3, 1), dtype=np.int64)
    indices_out = np.empty((3, 1), dtype=np.int64)
    out = (values_out, indices_out)
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    # 10
    input = np.array([[3.2, -0.1, 7.7, -8.8, 1.1],
                      [0.0, 0.0, 0.0, 0.0, -0.0]], dtype=np.float32)
    dim = 1
    keepdim = True
    values_out = np.empty((2, 1), dtype=np.float32)
    indices_out = np.empty((2, 1), dtype=np.int64)
    out = (values_out, indices_out)
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    # 11
    input = np.array([[[1.0, -2.0, 3.0, -4.0, 5.0],
                       [6.0, 7.0, -8.0, 9.0, 10.0],
                       [-1.0, -1.0, -1.0, -1.0, -1.0],
                       [2.2, 2.1, 2.0, 1.9, 1.8]],
                      [[-9.0, -8.0, -7.0, -6.0, -5.0],
                       [4.4, 4.3, 4.2, 4.1, 4.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [7.7, 7.6, 7.5, 7.4, 7.3]],
                      [[-1.5, -1.6, -1.7, -1.8, -1.9],
                       [3.3, 3.2, 3.1, 3.0, 2.9],
                       [8.8, 8.7, 8.6, 8.5, 8.4],
                       [-0.1, -0.2, -0.3, -0.4, -0.5]]], dtype=np.float32)
    dim = -3
    keepdim = False
    values_out = np.empty((4, 5), dtype=np.float32)
    indices_out = np.empty((4, 5), dtype=np.int64)
    out = (values_out, indices_out)
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    # 12
    input = np.array([[[[1, 2],
                        [3, 4]],
                       [[-1, -2],
                        [-3, -4]]],
                      [[[10, 9],
                        [8, 7]],
                       [[6, 5],
                        [4, 3]]]], dtype=np.int16)
    dim = 2
    keepdim = True
    values_out = np.empty((2, 2, 1, 2), dtype=np.int16)
    indices_out = np.empty((2, 2, 1, 2), dtype=np.int64)
    out = (values_out, indices_out)
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    return list_of_inputs

generated_inputs["torch.min_2"] = min_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.min_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.min_2'.")


check_valid('torch.min', generated_inputs['torch.min_2'], lib="torch", suffix=2)
