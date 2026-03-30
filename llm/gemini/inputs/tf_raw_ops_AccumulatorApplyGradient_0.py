
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_AccumulatorApplyGradient_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.constant("handle1", dtype=tf.string)
    local_step = np.array(1, dtype=np.int64)
    gradient = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.constant("handle2", dtype=tf.string)
    local_step = np.array(2, dtype=np.int64)
    gradient = np.array([[1, 2], [3, 4]], dtype=np.int32)

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.constant("handle3", dtype=tf.string)
    local_step = np.array(3, dtype=np.int64)
    gradient = np.array([-1.0, -2.0], dtype=np.float64)

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.constant("handle4", dtype=tf.string)
    local_step = np.array(4, dtype=np.int64)
    gradient = np.array([1, 2, 3, 4], dtype=np.int64)

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.constant("handle5", dtype=tf.string)
    local_step = np.array(5, dtype=np.int64)
    gradient = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.constant("handle6", dtype=tf.string)
    local_step = np.array(6, dtype=np.int64)
    gradient = np.array([1.0 + 1.0j, 2.0 + 2.0j], dtype=np.complex128)

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.constant("handle7", dtype=tf.string)
    local_step = np.array(7, dtype=np.int64)
    gradient = np.array([1, 2, 3], dtype=np.int16)

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.constant("handle8", dtype=tf.string)
    local_step = np.array(8, dtype=np.int64)
    gradient = np.array([1, 2, 3], dtype=np.uint8)

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.constant("handle9", dtype=tf.string)
    local_step = np.array(9, dtype=np.int64)
    gradient = np.array([1, 2, 3], dtype=np.uint32)

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = tf.constant("handle10", dtype=tf.string)
    local_step = np.array(10, dtype=np.int64)
    gradient = np.array([1, 2, 3], dtype=np.uint64)

    input_dict = {
        "handle": handle,
        "local_step": local_step,
        "gradient": gradient,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AccumulatorApplyGradient"] = tf_raw_ops_AccumulatorApplyGradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulatorApplyGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorApplyGradient'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.AccumulatorApplyGradient', generated_inputs['tf.raw_ops.AccumulatorApplyGradient'], lib="tf", suffix=0)
