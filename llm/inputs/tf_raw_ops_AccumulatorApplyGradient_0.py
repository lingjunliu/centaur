
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_accumulator_apply_gradient_inputs():
    list_of_inputs = []

    # Input 1
    handle = tf.Variable(b"test_accumulator_1", dtype=tf.string)
    local_step = np.array(1, dtype=np.int64)
    gradient = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_grad_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = tf.Variable(b"test_accumulator_2", dtype=tf.string)
    local_step = np.array(2, dtype=np.int64)
    gradient = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_grad_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = tf.Variable(b"test_accumulator_3", dtype=tf.string)
    local_step = np.array(100, dtype=np.int64)
    gradient = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_grad_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = tf.Variable(b"test_accumulator_4", dtype=tf.string)
    local_step = np.array(-5, dtype=np.int64)
    gradient = np.array([1, 2, 3], dtype=np.int64)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_grad_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = tf.Variable(b"test_accumulator_5", dtype=tf.string)
    local_step = np.array(0, dtype=np.int64)
    gradient = np.array([1.0], dtype=np.float32)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_grad_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = tf.Variable(b"test_accumulator_6", dtype=tf.string)
    local_step = np.array(2**7, dtype=np.int64)
    gradient = np.array([1, 2, 3, 4], dtype=np.uint8)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_grad_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = tf.Variable(b"test_accumulator_7", dtype=tf.string)
    local_step = np.array(-1, dtype=np.int64)
    gradient = np.array([1, 2, 3, 4], dtype=np.int16)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_grad_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = tf.Variable(b"test_accumulator_8", dtype=tf.string)
    local_step = np.array(2, dtype=np.int64)
    gradient = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_grad_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = tf.Variable(b"test_accumulator_9", dtype=tf.string)
    local_step = np.array(10, dtype=np.int64)
    gradient = np.array(5, dtype=np.int32)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_grad_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    handle = tf.Variable(b"test_accumulator_10", dtype=tf.string)
    local_step = np.array(10, dtype=np.int64)
    gradient = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_grad_10"}
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
