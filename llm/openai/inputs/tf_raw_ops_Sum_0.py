
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import torch
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_sum_inputs():
    list_of_inputs = []

    # Input 1
    inp = np.array([1, 2, 3], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    keep_dims = False
    name = "sum_int32_1d_axis0"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 2
    inp = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    axis = np.array(1, dtype=np.int64)
    keep_dims = True
    name = "sum_float32_2d_axis1_keep"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 3
    inp = np.array([[[1, -1, 2], [3, -3, 4]]], dtype=np.int64)
    axis = np.array([-1], dtype=np.int32)
    keep_dims = False
    name = "sum_int64_3d_last_axis"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 4
    inp = np.array([[1 + 2j, -3 + 0.5j], [4 - 1j, -2 - 2j]], dtype=np.complex64)
    axis = np.array([0, 1], dtype=np.int64)
    keep_dims = False
    name = "sum_complex64_all_axes"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 5
    inp = np.arange(2 * 1 * 3 * 4, dtype=np.float64).reshape(2, 1, 3, 4)
    axis = np.array(-2, dtype=np.int32)
    keep_dims = True
    name = "sum_float64_4d_negaxis_keep"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 6
    inp = np.array([-5, 0, 5, 10, -10], dtype=np.int16)
    axis = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "sum_int16_vector_axis0"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 7
    inp = np.array([[[1], [2], [3]], [[4], [5], [6]]], dtype=np.float16)
    axis = np.array([0, 1, 2], dtype=np.int32)
    keep_dims = True
    name = "sum_float16_reduce_all_keep"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 8
    inp = np.array([[1 + 0j, 2 - 1j, -3 + 2j], [0 + 0j, -1 - 1j, 4 + 0j]], dtype=np.complex128)
    axis = np.array(-1, dtype=np.int64)
    keep_dims = False
    name = "sum_complex128_last_axis"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 9
    inp = np.empty((2, 0, 3), dtype=np.int8)
    axis = np.array([-2], dtype=np.int64)
    keep_dims = True
    name = "sum_int8_zero_len_axis_keep"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 10
    inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = np.array([], dtype=np.int32)
    keep_dims = False
    name = "sum_int32_empty_axis_noop"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 11
    inp = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    axis = np.array([0, 2, 3], dtype=np.int32)
    keep_dims = False
    name = "sum_float32_4d_axes_0_2_3"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    return list_of_inputs

generated_inputs["tf.raw_ops.Sum"] = tf_raw_ops_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Sum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Sum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Sum', generated_inputs['tf.raw_ops.Sum'], lib="tf", suffix=0)
