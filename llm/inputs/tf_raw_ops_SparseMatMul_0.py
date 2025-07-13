
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparse_mat_mul_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    transpose_a = False
    transpose_b = False
    a_is_sparse = False
    b_is_sparse = False
    name = "matmul_1"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "a_is_sparse": a_is_sparse,
        "b_is_sparse": b_is_sparse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 0.0], [0.0, 8.0]], dtype=np.float32)
    transpose_a = True
    transpose_b = True
    a_is_sparse = True
    b_is_sparse = True
    name = "matmul_2"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "a_is_sparse": a_is_sparse,
        "b_is_sparse": b_is_sparse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    b = np.array([[4.0], [5.0], [6.0]], dtype=np.float32)
    transpose_a = False
    transpose_b = False
    a_is_sparse = False
    b_is_sparse = False
    name = "matmul_3"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "a_is_sparse": a_is_sparse,
        "b_is_sparse": b_is_sparse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    transpose_a = False
    transpose_b = False
    a_is_sparse = False
    b_is_sparse = False
    name = "matmul_4"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "a_is_sparse": a_is_sparse,
        "b_is_sparse": b_is_sparse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    b = np.array([[5.0, 0.0], [0.0, 8.0]], dtype=np.float32)
    transpose_a = False
    transpose_b = False
    a_is_sparse = True
    b_is_sparse = False
    name = "matmul_5"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "a_is_sparse": a_is_sparse,
        "b_is_sparse": b_is_sparse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    a = np.array([[1.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 0.0], [0.0, 8.0]], dtype=np.float32)
    transpose_a = False
    transpose_b = False
    a_is_sparse = False
    b_is_sparse = True
    name = "matmul_6"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "a_is_sparse": a_is_sparse,
        "b_is_sparse": b_is_sparse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[3.0], [4.0]], dtype=np.float32)
    transpose_a = True
    transpose_b = False
    a_is_sparse = False
    b_is_sparse = False
    name = "matmul_7"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "a_is_sparse": a_is_sparse,
        "b_is_sparse": b_is_sparse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[3.0], [4.0]], dtype=np.float32)
    transpose_a = False
    transpose_b = True
    a_is_sparse = False
    b_is_sparse = False
    name = "matmul_8"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "a_is_sparse": a_is_sparse,
        "b_is_sparse": b_is_sparse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    transpose_a = True
    transpose_b = False
    a_is_sparse = False
    b_is_sparse = False
    name = "matmul_9"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "a_is_sparse": a_is_sparse,
        "b_is_sparse": b_is_sparse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    transpose_a = False
    transpose_b = True
    a_is_sparse = False
    b_is_sparse = False
    name = "matmul_10"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "a_is_sparse": a_is_sparse,
        "b_is_sparse": b_is_sparse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_sparse_mat_mul_inputs()
generated_inputs["tf.raw_ops.SparseMatMul"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.SparseMatMul"].append(input_dict)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseMatMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseMatMul'.")

check_valid('tf.raw_ops.SparseMatMul', generated_inputs['tf.raw_ops.SparseMatMul'], lib="tf", suffix=0)
