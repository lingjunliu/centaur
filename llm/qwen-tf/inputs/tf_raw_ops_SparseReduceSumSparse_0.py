
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseReduceSumSparse_inputs():
    list_of_inputs = []

    input_indices = np.array([[0, 1], [2, 3], [1, 0]], dtype=np.int64)
    input_values = np.array([1.0, 2.5, -3.0], dtype=np.float32)
    input_shape = np.array([3, 4], dtype=np.int64)
    reduction_axes = np.array([1], dtype=np.int32)
    keep_dims = False
    name = "srs_case1"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 0], [0, 0], [1, 2], [2, 1]], dtype=np.int64)
    input_values = np.array([5, -2, 7, 3], dtype=np.int64)
    input_shape = np.array([3, 4], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = True
    name = "srs_case2"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[1, 2, 3], [0, 0, 0], [1, 0, 2], [0, 2, 1], [1, 1, 1]], dtype=np.int64)
    input_values = np.array([1, 2, -1, 3, 4], dtype=np.int32)
    input_shape = np.array([2, 3, 4], dtype=np.int64)
    reduction_axes = np.array([-1], dtype=np.int32)
    keep_dims = False
    name = "srs_case3"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 1, 2], [1, 2, 3], [1, 0, 0], [0, 2, 1]], dtype=np.int64)
    input_values = np.array([1+2j, -3+0.5j, 2-1j, -0-1j], dtype=np.complex64)
    input_shape = np.array([2, 3, 4], dtype=np.int64)
    reduction_axes = np.array([0, 2], dtype=np.int32)
    keep_dims = True
    name = "srs_case4"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    input_values = np.array([127, -1], dtype=np.int8)
    input_shape = np.array([2, 2], dtype=np.int64)
    reduction_axes = np.array([], dtype=np.int32)
    keep_dims = False
    name = "srs_case5"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0], [2], [4]], dtype=np.int64)
    input_values = np.array([-1.5, 2.0, 3.25], dtype=np.float64)
    input_shape = np.array([5], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = True
    name = "srs_case6"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([
        [0, 0, 0, 0],
        [1, 0, 2, 1],
        [1, 0, 1, 0],
        [0, 0, 2, 1],
        [0, 0, 1, 1],
        [1, 0, 2, 0]
    ], dtype=np.int64)
    input_values = np.array([0.5, -1.0, 3.0, 2.0, -0.5, 4.0], dtype=np.float64)
    input_shape = np.array([2, 1, 3, 2], dtype=np.int64)
    reduction_axes = np.array([1, 2], dtype=np.int32)
    keep_dims = False
    name = "srs_case7"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=np.int64)
    input_values = np.array([10, 20, 30], dtype=np.int32)
    input_shape = np.array([2, 2, 2], dtype=np.int64)
    reduction_axes = np.array([-3, -2, -1], dtype=np.int32)
    keep_dims = True
    name = "srs_case8"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 1], [1, 2], [2, 0], [2, 2]], dtype=np.int64)
    input_values = np.array([1+1j, -2+0j, 0+3j, 4-1j], dtype=np.complex64)
    input_shape = np.array([3, 3], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = True
    name = "srs_case9"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.empty((0, 2), dtype=np.int64)
    input_values = np.array([], dtype=np.int8)
    input_shape = np.array([4, 3], dtype=np.int64)
    reduction_axes = np.array([1], dtype=np.int32)
    keep_dims = False
    name = "srs_case10"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 0, 0], [1, 2, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int64)
    input_values = np.array([1.0, -2.0, 3.5, -0.5], dtype=np.float32)
    input_shape = np.array([2, 3, 1], dtype=np.int64)
    reduction_axes = np.array([2], dtype=np.int32)
    keep_dims = True
    name = "srs_case11"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0]], dtype=np.int64)
    input_values = np.array([123], dtype=np.int32)
    input_shape = np.array([1], dtype=np.int64)
    reduction_axes = np.array([], dtype=np.int32)
    keep_dims = True
    name = "srs_case12"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    input_values = np.array([100, 200], dtype=np.int32)
    input_shape = np.array([2, 3], dtype=np.int64)
    reduction_axes = np.array([0, 1], dtype=np.int32)
    keep_dims = False
    name = "srs_case13"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 0, 1], [2, 1, 0], [1, 1, 1]], dtype=np.int64)
    input_values = np.array([300, -200, 100], dtype=np.int32)
    input_shape = np.array([3, 2, 2], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "srs_case14"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseReduceSumSparse"] = tf_raw_ops_SparseReduceSumSparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseReduceSumSparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseReduceSumSparse'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseReduceSumSparse', generated_inputs['tf.raw_ops.SparseReduceSumSparse'], lib="tf", suffix=0)
