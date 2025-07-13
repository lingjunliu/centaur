
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_matmul_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    transpose_a = False
    transpose_b = False
    grad_a = False
    grad_b = False
    name = "matmul_1"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "grad_a": grad_a,
        "grad_b": grad_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1, 2, 3]], dtype=np.float32)
    b = np.array([[4], [5], [6]], dtype=np.float32)
    transpose_a = False
    transpose_b = False
    grad_a = False
    grad_b = False
    name = "matmul_2"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "grad_a": grad_a,
        "grad_b": grad_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    transpose_a = True
    transpose_b = True
    grad_a = True
    grad_b = True
    name = "matmul_3"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "grad_a": grad_a,
        "grad_b": grad_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[1j, 2j], [3j, 4j]], dtype=np.complex64)
    b = np.array([[5j, 6j], [7j, 8j]], dtype=np.complex64)
    transpose_a = False
    transpose_b = False
    grad_a = False
    grad_b = False
    name = "matmul_4"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "grad_a": grad_a,
        "grad_b": grad_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([[5, 6], [7, 8]], dtype=np.int32)
    transpose_a = False
    transpose_b = False
    grad_a = False
    grad_b = False
    name = "matmul_5"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "grad_a": grad_a,
        "grad_b": grad_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([[1, 2], [3, 4]], dtype=np.float64)
    b = np.array([[5, 6], [7, 8]], dtype=np.float64)
    transpose_a = True
    transpose_b = False
    grad_a = False
    grad_b = False
    name = "matmul_6"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "grad_a": grad_a,
        "grad_b": grad_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    b = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    transpose_a = False
    transpose_b = True
    grad_a = False
    grad_b = False
    name = "matmul_7"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "grad_a": grad_a,
        "grad_b": grad_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([[1, 2], [3, 4]], dtype=np.int64)
    b = np.array([[5, 6], [7, 8]], dtype=np.int64)
    transpose_a = True
    transpose_b = True
    grad_a = True
    grad_b = False
    name = "matmul_8"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "grad_a": grad_a,
        "grad_b": grad_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    b = np.array([[5.5, 6.5], [7.5, 8.5]], dtype=np.float32)
    transpose_a = False
    transpose_b = False
    grad_a = False
    grad_b = True
    name = "matmul_9"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "grad_a": grad_a,
        "grad_b": grad_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.int32)
    transpose_a = False
    transpose_b = False
    grad_a = False
    grad_b = False
    name = "matmul_10"

    input_dict = {
        "a": a,
        "b": b,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "grad_a": grad_a,
        "grad_b": grad_b,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_matmul_inputs()
generated_inputs["tf.raw_ops.MatMul"] = []
for input_dict in inputs:
  a = tf.convert_to_tensor(input_dict['a'])
  b = tf.convert_to_tensor(input_dict['b'])
  transpose_a = input_dict['transpose_a']
  transpose_b = input_dict['transpose_b']
  grad_a = input_dict['grad_a']
  grad_b = input_dict['grad_b']
  name = input_dict['name']

  generated_inputs["tf.raw_ops.MatMul"].append({
            "a": a,
            "b": b,
            "transpose_a": transpose_a,
            "transpose_b": transpose_b,
            "grad_a": grad_a,
            "grad_b": grad_b,
            "name": name
        })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatMul'.")

check_valid('tf.raw_ops.MatMul', generated_inputs['tf.raw_ops.MatMul'], lib="tf", suffix=0)
