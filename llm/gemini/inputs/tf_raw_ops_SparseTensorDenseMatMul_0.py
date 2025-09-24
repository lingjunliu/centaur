
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseTensorDenseMatMul_inputs():
    list_of_inputs = []

    # Input 1
    a_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    a_values = np.array([1.0, 2.0], dtype=np.float32)
    a_shape = np.array([2, 3], dtype=np.int64)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    adjoint_a = False
    adjoint_b = False
    name = "test_matmul_1"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a_indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    a_values = np.array([3, 4], dtype=np.int32)
    a_shape = np.array([2, 2], dtype=np.int64)
    b = np.array([[5, 6], [7, 8]], dtype=np.int32)
    adjoint_a = True
    adjoint_b = False
    name = "test_matmul_2"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a_indices = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int64)
    a_values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    a_shape = np.array([2, 2], dtype=np.int64)
    b = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float64)
    adjoint_a = False
    adjoint_b = True
    name = "test_matmul_3"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    a_values = np.array([1+1j, 2+2j], dtype=np.complex64)
    a_shape = np.array([2, 2], dtype=np.int64)
    b = np.array([[3+3j, 4+4j], [5+5j, 6+6j]], dtype=np.complex64)
    adjoint_a = True
    adjoint_b = True
    name = "test_matmul_4"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a_indices = np.array([[0, 0], [2, 1]], dtype=np.int64)
    a_values = np.array([1, 2], dtype=np.int32)
    a_shape = np.array([3, 2], dtype=np.int64)
    b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    adjoint_a = False
    adjoint_b = False
    name = "test_matmul_5"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    a_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    a_values = np.array([1.0, -2.0], dtype=np.float32)
    a_shape = np.array([2, 3], dtype=np.int64)
    b = np.array([[-1.0, 2.0], [3.0, -4.0], [5.0, 6.0]], dtype=np.float32)
    adjoint_a = False
    adjoint_b = False
    name = "test_matmul_6"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a_indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    a_values = np.array([-3, 4], dtype=np.int32)
    a_shape = np.array([2, 2], dtype=np.int64)
    b = np.array([[5, -6], [-7, 8]], dtype=np.int32)
    adjoint_a = True
    adjoint_b = False
    name = "test_matmul_7"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a_indices = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int64)
    a_values = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    a_shape = np.array([2, 2], dtype=np.int64)
    b = np.array([[-4.0, 5.0], [6.0, -7.0]], dtype=np.float64)
    adjoint_a = False
    adjoint_b = True
    name = "test_matmul_8"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    a_values = np.array([-1-1j, 2+2j], dtype=np.complex64)
    a_shape = np.array([2, 2], dtype=np.int64)
    b = np.array([[3+3j, -4-4j], [-5-5j, 6+6j]], dtype=np.complex64)
    adjoint_a = True
    adjoint_b = True
    name = "test_matmul_9"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a_indices = np.array([[0, 0], [2, 1]], dtype=np.int64)
    a_values = np.array([1, -2], dtype=np.int32)
    a_shape = np.array([3, 2], dtype=np.int64)
    b = np.array([[-1, 2, 3], [4, -5, 6]], dtype=np.int32)
    adjoint_a = False
    adjoint_b = False
    name = "test_matmul_10"

    input_dict = {
        "a_indices": a_indices,
        "a_values": a_values,
        "a_shape": a_shape,
        "b": b,
        "adjoint_a": adjoint_a,
        "adjoint_b": adjoint_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseTensorDenseMatMul"] = tf_raw_ops_SparseTensorDenseMatMul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseTensorDenseMatMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseTensorDenseMatMul'.")

check_valid('tf.raw_ops.SparseTensorDenseMatMul', generated_inputs['tf.raw_ops.SparseTensorDenseMatMul'], lib="tf", suffix=0)
