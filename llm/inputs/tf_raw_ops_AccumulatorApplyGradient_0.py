
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_accumulator_apply_gradient_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("accumulator_handle", dtype=tf.string)
    local_step = tf.constant(1, dtype=tf.int64)
    gradient = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    name = "apply_gradient_1"

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("another_handle", dtype=tf.string)
    local_step = tf.constant(2, dtype=tf.int64)
    gradient = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    name = "apply_gradient_2"

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("handle_3", dtype=tf.string)
    local_step = tf.constant(3, dtype=tf.int64)
    gradient = tf.constant(np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64))
    name = "apply_gradient_3"

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("handle_4", dtype=tf.string)
    local_step = tf.constant(4, dtype=tf.int64)
    gradient = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    name = "apply_gradient_4"

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("handle_5", dtype=tf.string)
    local_step = tf.constant(5, dtype=tf.int64)
    gradient = tf.constant(np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64))
    name = "apply_gradient_5"

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("handle_6", dtype=tf.string)
    local_step = tf.constant(6, dtype=tf.int64)
    gradient = tf.constant(np.array([[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]], dtype=np.complex128))
    name = "apply_gradient_6"

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("handle_7", dtype=tf.string)
    local_step = tf.constant(7, dtype=tf.int64)
    gradient = tf.constant(np.array([1, 2, 3], dtype=np.int8))
    name = "apply_gradient_7"

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("handle_8", dtype=tf.string)
    local_step = tf.constant(8, dtype=tf.int64)
    gradient = tf.constant(np.array([1, 2, 3, 4], dtype=np.int16))
    name = "apply_gradient_8"

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("handle_9", dtype=tf.string)
    local_step = tf.constant(9, dtype=tf.int64)
    gradient = tf.constant(np.array([1.0, 2.0], dtype=np.float32).reshape((2,1)), name="gradient")
    name = "apply_gradient_9"

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    handle = tf.constant("handle_10", dtype=tf.string)
    local_step = tf.constant(10, dtype=tf.int64)
    gradient = tf.constant(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2,2)), name="gradient")
    name = "apply_gradient_10"

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AccumulatorApplyGradient"] = tf_raw_ops_accumulator_apply_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AccumulatorApplyGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorApplyGradient'.")

check_valid('tf.raw_ops.AccumulatorApplyGradient', generated_inputs['tf.raw_ops.AccumulatorApplyGradient'], lib="tf", suffix=0)
