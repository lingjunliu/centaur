
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def gt_inputs():
    list_of_inputs = []

    # 1
    input_arr = np.array([1.0, 2.0, -3.0], dtype=np.float32)
    other_arr = np.array([0.0, 5.0, -3.0], dtype=np.float32)
    out_arr = np.empty((3,), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 2
    input_arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    other_arr = np.array([[1, 0], [4, 4]], dtype=np.int64)
    out_arr = np.empty((2, 2), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 3
    input_arr = np.array([[-5, 0, 7], [12, -3, 4]], dtype=np.int16)
    other_arr = np.array([[0.5, -1.5, 6.5]], dtype=np.float32)
    out_arr = np.empty((2, 3), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 4
    input_arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    other_arr = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float64)
    out_arr = np.empty((2, 3, 4), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 5
    input_arr = np.array([[1.2, 1.8], [-0.5, 2.3]], dtype=np.float64)
    other_arr = np.array(1.5, dtype=np.float64)
    out_arr = np.empty((2, 2), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 6
    input_arr = np.array([[True, False, True], [False, False, True]], dtype=bool)
    other_arr = np.array([[False, False, True], [True, False, False]], dtype=bool)
    out_arr = np.empty((2, 3), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 7
    input_arr = np.array([[np.nan, np.inf, -np.inf], [0.0, -1.0, 2.0]], dtype=np.float32)
    other_arr = np.array([[0.0, np.inf, -np.inf], [np.nan, -2.0, 2.0]], dtype=np.float32)
    out_arr = np.empty((2, 3), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 8
    input_arr = np.array([[[1, 200, 3]], [[4, 5, 6]]], dtype=np.uint8)  # shape (2,1,3)
    other_arr = np.array([[0, 100, 10]], dtype=np.uint8)                # shape (1,3)
    out_arr = np.empty((2, 1, 3), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 9
    input_arr = np.empty((0, 3), dtype=np.float32)
    other_arr = np.empty((0, 3), dtype=np.float32)
    out_arr = np.empty((0, 3), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 10
    input_arr = np.array([[[[1], [2], [3]]], [[[4], [5], [6]]]], dtype=np.int32)  # shape (2,1,3,1)
    other_arr = np.array([[1, 2, 3]], dtype=np.int32)                              # shape (1,3,1) after expand
    other_arr = other_arr.reshape(1, 3, 1)
    out_arr = np.empty((2, 1, 3, 1), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 11
    input_arr = np.array([-2, 0, 2, 4, 6], dtype=np.int8)          # shape (5,)
    other_arr = np.array([[-3], [0], [3], [6], [9]], dtype=np.int8)  # shape (5,1)
    out_arr = np.empty((5, 5), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 12
    input_arr = np.array(3.0, dtype=np.float16)                    # shape ()
    other_arr = np.array([[1.0, 4.0], [3.0, 2.0]], dtype=np.float16)  # shape (2,2)
    out_arr = np.empty((2, 2), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.gt_1"] = gt_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.gt_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gt_1'.")


check_valid('torch.gt', generated_inputs['torch.gt_1'], lib="torch", suffix=1)
