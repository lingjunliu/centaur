
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseReduceMax_inputs():
    list_of_inputs = []

    # Input 1
    input_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    input_values = np.array([1.0, 2.0], dtype=np.float32)
    input_shape = np.array([2, 3], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "sparse_reduce_max_1"

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    input_values = np.array([1, 2, 3, 4], dtype=np.int32)
    input_shape = np.array([2, 2], dtype=np.int64)
    reduction_axes = np.array([0, 1], dtype=np.int32)
    keep_dims = True
    name = "sparse_reduce_max_2"

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
    input_values = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    input_shape = np.array([2, 2, 2], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "sparse_reduce_max_3"

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    input_values = np.array([-1, -2, -3], dtype=np.int32)
    input_shape = np.array([3, 3], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = True
    name = "sparse_reduce_max_4"

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    input_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    input_values = np.array([1, 2, 3, 4], dtype=np.uint8)
    input_shape = np.array([2, 2], dtype=np.int64)
    reduction_axes = np.array([1], dtype=np.int32)
    keep_dims = False
    name = "sparse_reduce_max_5"

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_indices = np.array([[0], [1]], dtype=np.int64)
    input_values = np.array([1, 2], dtype=np.int16)
    input_shape = np.array([2], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "sparse_reduce_max_6"

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Removed bfloat16 as numpy doesn't support it natively
    input_indices = np.array([[0,0,0]], dtype=np.int64)
    input_values = np.array([1.0], dtype=np.float16)
    input_shape = np.array([1,1,1], dtype=np.int64)
    reduction_axes = np.array([0,1,2], dtype=np.int32)
    keep_dims = True
    name = "sparse_reduce_max_7"

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - Removed uint16 as it causes ValueError
    # input_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    # input_values = np.array([1, 2], dtype=np.uint16)
    # input_shape = np.array([2, 3], dtype=np.int64)
    # reduction_axes = np.array([1], dtype=np.int32)
    # keep_dims = True
    # name = "sparse_reduce_max_8"

    # input_dict = {
    #     "input_indices": input_indices,
    #     "input_values": input_values,
    #     "input_shape": input_shape,
    #     "reduction_axes": reduction_axes,
    #     "keep_dims": keep_dims,
    #     "name": name
    # }
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    input_values = np.array([1.0, 2.0], dtype=np.float16)
    input_shape = np.array([2, 3], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "sparse_reduce_max_9"

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - Removed uint32, as it causes a value error
    # input_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    # input_values = np.array([1, 2], dtype=np.uint32)
    # input_shape = np.array([2, 3], dtype=np.int64)
    # reduction_axes = np.array([0], dtype=np.int32)
    # keep_dims = True
    # name = "sparse_reduce_max_10"

    # input_dict = {
    #     "input_indices": input_indices,
    #     "input_values": input_values,
    #     "input_shape": input_shape,
    #     "reduction_axes": reduction_axes,
    #     "keep_dims": keep_dims,
    #     "name": name
    # }
    # list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11 - uint64
    input_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    input_values = np.array([1, 2], dtype=np.int64)
    input_shape = np.array([2, 3], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = True
    name = "sparse_reduce_max_11"

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseReduceMax"] = tf_raw_ops_SparseReduceMax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseReduceMax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseReduceMax'.")

check_valid('tf.raw_ops.SparseReduceMax', generated_inputs['tf.raw_ops.SparseReduceMax'], lib="tf", suffix=0)
