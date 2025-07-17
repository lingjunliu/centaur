
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseAccumulatorApplyGradient_inputs():
    list_of_inputs = []

    # Input 1
    handle = np.array(b"accumulator_handle_1")
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
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = np.array(b"accumulator_handle_2")
    local_step = np.array(2, dtype=np.int64)
    gradient_indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    gradient_values = np.array([3, 4], dtype=np.int32)
    gradient_shape = np.array([2, 2, 2], dtype=np.int64)
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
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = np.array(b"accumulator_handle_3")
    local_step = np.array(3, dtype=np.int64)
    gradient_indices = np.array([[0], [1]], dtype=np.int64)
    gradient_values = np.array([5.5, 6.6], dtype=np.float64)
    gradient_shape = np.array([5], dtype=np.int64)
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
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = np.array(b"accumulator_handle_4")
    local_step = np.array(4, dtype=np.int64)
    gradient_indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
    gradient_values = np.array([7, 8], dtype=np.int64)
    gradient_shape = np.array([4, 4], dtype=np.int64)
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
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = np.array(b"accumulator_handle_5")
    local_step = np.array(5, dtype=np.int64)
    gradient_indices = np.array([[0, 0], [1, 1], [2,2]], dtype=np.int64)
    gradient_values = np.array([9, 10, 11], dtype=np.uint8)
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
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = np.array(b"accumulator_handle_6")
    local_step = np.array(6, dtype=np.int64)
    gradient_indices = np.array([[0], [1]], dtype=np.int64)
    gradient_values = np.array([12, 13], dtype=np.int16)
    gradient_shape = np.array([2], dtype=np.int64)
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
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = np.array(b"accumulator_handle_7")
    local_step = np.array(7, dtype=np.int64)
    gradient_indices = np.array([[0,0,0], [0,0,1]], dtype=np.int64)
    gradient_values = np.array([14, 15], dtype=np.int8)
    gradient_shape = np.array([1, 1, 2], dtype=np.int64)
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
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = np.array(b"accumulator_handle_8")
    local_step = np.array(8, dtype=np.int64)
    gradient_indices = np.array([[0], [1]], dtype=np.int64)
    gradient_values = np.array([complex(1, 2), complex(3, 4)], dtype=np.complex64)
    gradient_shape = np.array([2], dtype=np.int64)
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
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    handle = np.array(b"accumulator_handle_9")
    local_step = np.array(9, dtype=np.int64)
    gradient_indices = np.array([[0], [1]], dtype=np.int64)
    gradient_values = np.array([12, 13], dtype=np.float16)
    gradient_shape = np.array([2], dtype=np.int64)
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
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    handle = np.array(b"accumulator_handle_10")
    local_step = np.array(10, dtype=np.int64)
    gradient_indices = np.array([[0], [1]], dtype=np.int64)
    gradient_values = np.array([12, 13], dtype=np.uint16)
    gradient_shape = np.array([2], dtype=np.int64)
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
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseAccumulatorApplyGradient"] = tf_raw_ops_SparseAccumulatorApplyGradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseAccumulatorApplyGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseAccumulatorApplyGradient'.")

check_valid('tf.raw_ops.SparseAccumulatorApplyGradient', generated_inputs['tf.raw_ops.SparseAccumulatorApplyGradient'], lib="tf", suffix=0)
