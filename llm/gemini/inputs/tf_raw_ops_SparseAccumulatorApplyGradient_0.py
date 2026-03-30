
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparse_accumulator_apply_gradient_inputs():
    list_of_inputs = []

    # Input 1
    handle = np.array(b"accumulator_handle", dtype=np.string_)
    local_step = np.array(1, dtype=np.int64)
    gradient_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    gradient_values = np.array([1.0, 2.0], dtype=np.float32)
    gradient_shape = np.array([2, 3], dtype=np.int64)
    has_known_shape = True

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient_indices": gradient_indices,
        "gradient_values": gradient_values,
        "gradient_shape": gradient_shape,
        "has_known_shape": has_known_shape,
        "name": None
    }
    list_of_inputs.append(input_dict)

    # Input 2
    handle = np.array(b"another_handle", dtype=np.string_)
    local_step = np.array(2, dtype=np.int64)
    gradient_indices = np.array([[0], [1]], dtype=np.int64)
    gradient_values = np.array([3, 4], dtype=np.int32)
    gradient_shape = np.array([5], dtype=np.int64)
    has_known_shape = False

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient_indices": gradient_indices,
        "gradient_values": gradient_values,
        "gradient_shape": gradient_shape,
        "has_known_shape": has_known_shape,
        "name": None
    }
    list_of_inputs.append(input_dict)

    # Input 3
    handle = np.array(b"yet_another_handle", dtype=np.string_)
    local_step = np.array(3, dtype=np.int64)
    gradient_indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    gradient_values = np.array([5.5, 6.6], dtype=np.float64)
    gradient_shape = np.array([2, 2, 2], dtype=np.int64)
    has_known_shape = True

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient_indices": gradient_indices,
        "gradient_values": gradient_values,
        "gradient_shape": gradient_shape,
        "has_known_shape": has_known_shape,
        "name": None
    }
    list_of_inputs.append(input_dict)

    # Input 4
    handle = np.array(b"handle4", dtype=np.string_)
    local_step = np.array(4, dtype=np.int64)
    gradient_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    gradient_values = np.array([7, 8], dtype=np.int64)
    gradient_shape = np.array([2, 2], dtype=np.int64)
    has_known_shape = False

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient_indices": gradient_indices,
        "gradient_values": gradient_values,
        "gradient_shape": gradient_shape,
        "has_known_shape": has_known_shape,
        "name": None
    }
    list_of_inputs.append(input_dict)

    # Input 5
    handle = np.array(b"handle5", dtype=np.string_)
    local_step = np.array(5, dtype=np.int64)
    gradient_indices = np.array([[0]], dtype=np.int64)
    gradient_values = np.array([9], dtype=np.float32)
    gradient_shape = np.array([1], dtype=np.int64)
    has_known_shape = True

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient_indices": gradient_indices,
        "gradient_values": gradient_values,
        "gradient_shape": gradient_shape,
        "has_known_shape": has_known_shape,
        "name": None
    }
    list_of_inputs.append(input_dict)

   # Input 6
    handle = np.array(b"handle6", dtype=np.string_)
    local_step = np.array(6, dtype=np.int64)
    gradient_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    gradient_values = np.array([10, 11], dtype=np.float64)
    gradient_shape = np.array([1, 2], dtype=np.int64)
    has_known_shape = False

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient_indices": gradient_indices,
        "gradient_values": gradient_values,
        "gradient_shape": gradient_shape,
        "has_known_shape": has_known_shape,
        "name": None
    }
    list_of_inputs.append(input_dict)

    # Input 7
    handle = np.array(b"handle7", dtype=np.string_)
    local_step = np.array(7, dtype=np.int64)
    gradient_indices = np.array([[0, 0, 0]], dtype=np.int64)
    gradient_values = np.array([12], dtype=np.int32)
    gradient_shape = np.array([1, 1, 1], dtype=np.int64)
    has_known_shape = True

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient_indices": gradient_indices,
        "gradient_values": gradient_values,
        "gradient_shape": gradient_shape,
        "has_known_shape": has_known_shape,
        "name": None
    }
    list_of_inputs.append(input_dict)

    # Input 8
    handle = np.array(b"handle8", dtype=np.string_)
    local_step = np.array(8, dtype=np.int64)
    gradient_indices = np.array([[1, 0]], dtype=np.int64)
    gradient_values = np.array([13], dtype=np.int64)
    gradient_shape = np.array([2, 1], dtype=np.int64)
    has_known_shape = False

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient_indices": gradient_indices,
        "gradient_values": gradient_values,
        "gradient_shape": gradient_shape,
        "has_known_shape": has_known_shape,
        "name": None
    }
    list_of_inputs.append(input_dict)

    # Input 9
    handle = np.array(b"handle9", dtype=np.string_)
    local_step = np.array(9, dtype=np.int64)
    gradient_indices = np.array([[2, 2]], dtype=np.int64)
    gradient_values = np.array([14], dtype=np.uint8)
    gradient_shape = np.array([3, 3], dtype=np.int64)
    has_known_shape = True

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient_indices": gradient_indices,
        "gradient_values": gradient_values,
        "gradient_shape": gradient_shape,
        "has_known_shape": has_known_shape,
        "name": None
    }
    list_of_inputs.append(input_dict)

    # Input 10
    handle = np.array(b"handle10", dtype=np.string_)
    local_step = np.array(10, dtype=np.int64)
    gradient_indices = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
    gradient_values = np.array([15], dtype=np.float32)
    gradient_shape = np.array([1, 5], dtype=np.int64)
    has_known_shape = False

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient_indices": gradient_indices,
        "gradient_values": gradient_values,
        "gradient_shape": gradient_shape,
        "has_known_shape": has_known_shape,
        "name": None
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseAccumulatorApplyGradient"] = tf_raw_ops_sparse_accumulator_apply_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseAccumulatorApplyGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseAccumulatorApplyGradient'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseAccumulatorApplyGradient', generated_inputs['tf.raw_ops.SparseAccumulatorApplyGradient'], lib="tf", suffix=0)
