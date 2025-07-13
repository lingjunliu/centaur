
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def sparse_accumulator_apply_gradient_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.compat.as_bytes("test_handle_1")
    local_step = tf.constant(1, dtype=tf.int64)
    gradient_indices = tf.constant([[0, 0], [1, 2]], dtype=tf.int64)
    gradient_values = tf.constant([1.0, 2.0], dtype=tf.float32)
    gradient_shape = tf.constant([2, 3], dtype=tf.int64)
    has_known_shape = True
    input_dict = {'handle': handle, 'local_step': local_step, 'gradient_indices': gradient_indices,
                  'gradient_values': gradient_values, 'gradient_shape': gradient_shape,
                  'has_known_shape': has_known_shape, 'name': None}
    list_of_inputs.append(input_dict)

    # Input 2
    handle = tf.compat.as_bytes("test_handle_2")
    local_step = tf.constant(2, dtype=tf.int64)
    gradient_indices = tf.constant([[0, 1], [2, 0]], dtype=tf.int64)
    gradient_values = tf.constant([3, 4], dtype=tf.int32)
    gradient_shape = tf.constant([3, 2], dtype=tf.int64)
    has_known_shape = False
    input_dict = {'handle': handle, 'local_step': local_step, 'gradient_indices': gradient_indices,
                  'gradient_values': gradient_values, 'gradient_shape': gradient_shape,
                  'has_known_shape': has_known_shape, 'name': None}
    list_of_inputs.append(input_dict)

    # Input 3
    handle = tf.compat.as_bytes("test_handle_3")
    local_step = tf.constant(10, dtype=tf.int64)
    gradient_indices = tf.constant([[0, 0, 0], [1, 1, 1]], dtype=tf.int64)
    gradient_values = tf.constant([5.0, 6.0], dtype=tf.float64)
    gradient_shape = tf.constant([2, 2, 2], dtype=tf.int64)
    has_known_shape = True
    input_dict = {'handle': handle, 'local_step': local_step, 'gradient_indices': gradient_indices,
                  'gradient_values': gradient_values, 'gradient_shape': gradient_shape,
                  'has_known_shape': has_known_shape, 'name': None}
    list_of_inputs.append(input_dict)

   # Input 4
    handle = tf.compat.as_bytes("test_handle_4")
    local_step = tf.constant(5, dtype=tf.int64)
    gradient_indices = tf.constant([[0], [1]], dtype=tf.int64)
    gradient_values = tf.constant([7, 8], dtype=tf.int64)
    gradient_shape = tf.constant([2], dtype=tf.int64)
    has_known_shape = True
    input_dict = {'handle': handle, 'local_step': local_step, 'gradient_indices': gradient_indices,
                  'gradient_values': gradient_values, 'gradient_shape': gradient_shape,
                  'has_known_shape': has_known_shape, 'name': None}
    list_of_inputs.append(input_dict)

    # Input 5
    handle = tf.compat.as_bytes("test_handle_5")
    local_step = tf.constant(3, dtype=tf.int64)
    gradient_indices = tf.constant([[0, 0], [0, 1], [1,0], [1,1]], dtype=tf.int64)
    gradient_values = tf.constant([9, 10, 11, 12], dtype=tf.float32)
    gradient_shape = tf.constant([2, 2], dtype=tf.int64)
    has_known_shape = False
    input_dict = {'handle': handle, 'local_step': local_step, 'gradient_indices': gradient_indices,
                  'gradient_values': gradient_values, 'gradient_shape': gradient_shape,
                  'has_known_shape': has_known_shape, 'name': None}
    list_of_inputs.append(input_dict)

    # Input 6
    handle = tf.compat.as_bytes("test_handle_6")
    local_step = tf.constant(7, dtype=tf.int64)
    gradient_indices = tf.constant([[0,0,0], [0,0,1], [0,1,0]], dtype=tf.int64)
    gradient_values = tf.constant([13, 14, 15], dtype=tf.int32)
    gradient_shape = tf.constant([1, 2, 2], dtype=tf.int64)
    has_known_shape = True
    input_dict = {'handle': handle, 'local_step': local_step, 'gradient_indices': gradient_indices,
                  'gradient_values': gradient_values, 'gradient_shape': gradient_shape,
                  'has_known_shape': has_known_shape, 'name': None}
    list_of_inputs.append(input_dict)

    # Input 7
    handle = tf.compat.as_bytes("test_handle_7")
    local_step = tf.constant(9, dtype=tf.int64)
    gradient_indices = tf.constant([[1]], dtype=tf.int64)
    gradient_values = tf.constant([16], dtype=tf.float64)
    gradient_shape = tf.constant([2], dtype=tf.int64)
    has_known_shape = False
    input_dict = {'handle': handle, 'local_step': local_step, 'gradient_indices': gradient_indices,
                  'gradient_values': gradient_values, 'gradient_shape': gradient_shape,
                  'has_known_shape': has_known_shape, 'name': None}
    list_of_inputs.append(input_dict)

    # Input 8
    handle = tf.compat.as_bytes("test_handle_8")
    local_step = tf.constant(4, dtype=tf.int64)
    gradient_indices = tf.constant([[0,0],[1,1],[2,2]], dtype=tf.int64)
    gradient_values = tf.constant([1, 2, 3], dtype=tf.uint8)
    gradient_shape = tf.constant([3, 3], dtype=tf.int64)
    has_known_shape = True
    input_dict = {'handle': handle, 'local_step': local_step, 'gradient_indices': gradient_indices,
                  'gradient_values': gradient_values, 'gradient_shape': gradient_shape,
                  'has_known_shape': has_known_shape, 'name': None}
    list_of_inputs.append(input_dict)

    # Input 9
    handle = tf.compat.as_bytes("test_handle_9")
    local_step = tf.constant(6, dtype=tf.int64)
    gradient_indices = tf.constant([[0,0], [1,0]], dtype=tf.int64)
    gradient_values = tf.constant([4, 5], dtype=tf.complex64)
    gradient_shape = tf.constant([2, 1], dtype=tf.int64)
    has_known_shape = False
    input_dict = {'handle': handle, 'local_step': local_step, 'gradient_indices': gradient_indices,
                  'gradient_values': gradient_values, 'gradient_shape': gradient_shape,
                  'has_known_shape': has_known_shape, 'name': None}
    list_of_inputs.append(input_dict)

    # Input 10
    handle = tf.compat.as_bytes("test_handle_10")
    local_step = tf.constant(8, dtype=tf.int64)
    gradient_indices = tf.constant([[0,0,0],[1,1,1]], dtype=tf.int64)
    gradient_values = tf.constant([6, 7], dtype=tf.bfloat16)
    gradient_shape = tf.constant([2, 2, 2], dtype=tf.int64)
    has_known_shape = True
    input_dict = {'handle': handle, 'local_step': local_step, 'gradient_indices': gradient_indices,
                  'gradient_values': gradient_values, 'gradient_shape': gradient_shape,
                  'has_known_shape': has_known_shape, 'name': None}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseAccumulatorApplyGradient"] = sparse_accumulator_apply_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseAccumulatorApplyGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseAccumulatorApplyGradient'.")

check_valid('tf.raw_ops.SparseAccumulatorApplyGradient', generated_inputs['tf.raw_ops.SparseAccumulatorApplyGradient'], lib="tf", suffix=0)
