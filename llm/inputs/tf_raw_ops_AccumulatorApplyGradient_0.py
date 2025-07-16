
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AccumulatorApplyGradient_inputs():
    list_of_inputs = []

    # Input 1
    accumulator = tf.compat.v1.get_variable("accumulator", initializer=tf.zeros([3], dtype=tf.float32), use_resource=True, trainable=False)
    handle = tf.raw_ops.VarHandleOp(dtype=tf.float32, shape=tf.TensorShape([3]), shared_name="accumulator_handle", name="accumulator_handle", container="")
    local_step = tf.constant(1, dtype=tf.int64)
    gradient = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_gradient_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    accumulator = tf.compat.v1.get_variable("accumulator_2", initializer=tf.zeros([2, 2], dtype=tf.int32), use_resource=True, trainable=False)
    handle = tf.raw_ops.VarHandleOp(dtype=tf.int32, shape=tf.TensorShape([2, 2]), shared_name="accumulator_handle_2", name="accumulator_handle_2", container="")
    local_step = tf.constant(2, dtype=tf.int64)
    gradient = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_gradient_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    accumulator = tf.compat.v1.get_variable("accumulator_3", initializer=tf.zeros([4], dtype=tf.float64), use_resource=True, trainable=False)
    handle = tf.raw_ops.VarHandleOp(dtype=tf.float64, shape=tf.TensorShape([4]), shared_name="accumulator_handle_3", name="accumulator_handle_3", container="")
    local_step = tf.constant(3, dtype=tf.int64)
    gradient = tf.constant(np.array([1.5, 2.5, 3.5, 4.5]), dtype=tf.float64)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_gradient_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    accumulator = tf.compat.v1.get_variable("accumulator_4", initializer=tf.zeros([4], dtype=tf.uint8), use_resource=True, trainable=False)
    handle = tf.raw_ops.VarHandleOp(dtype=tf.uint8, shape=tf.TensorShape([4]), shared_name="accumulator_handle_4", name="accumulator_handle_4", container="")
    local_step = tf.constant(4, dtype=tf.int64)
    gradient = tf.constant(np.array([1, 2, 3, 4], dtype=np.uint8), dtype=tf.uint8)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_gradient_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    accumulator = tf.compat.v1.get_variable("accumulator_5", initializer=tf.zeros([4], dtype=tf.int16), use_resource=True, trainable=False)
    handle = tf.raw_ops.VarHandleOp(dtype=tf.int16, shape=tf.TensorShape([4]), shared_name="accumulator_handle_5", name="accumulator_handle_5", container="")
    local_step = tf.constant(5, dtype=tf.int64)
    gradient = tf.constant(np.array([1, 2, 3, 4], dtype=np.int16), dtype=tf.int16)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_gradient_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    accumulator = tf.compat.v1.get_variable("accumulator_6", initializer=tf.zeros([4], dtype=tf.int8), use_resource=True, trainable=False)
    handle = tf.raw_ops.VarHandleOp(dtype=tf.int8, shape=tf.TensorShape([4]), shared_name="accumulator_handle_6", name="accumulator_handle_6", container="")
    local_step = tf.constant(6, dtype=tf.int64)
    gradient = tf.constant(np.array([1, 2, 3, 4], dtype=np.int8), dtype=tf.int8)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_gradient_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    accumulator = tf.compat.v1.get_variable("accumulator_7", initializer=tf.zeros([4], dtype=tf.complex64), use_resource=True, trainable=False)
    handle = tf.raw_ops.VarHandleOp(dtype=tf.complex64, shape=tf.TensorShape([4]), shared_name="accumulator_handle_7", name="accumulator_handle_7", container="")
    local_step = tf.constant(7, dtype=tf.int64)
    gradient = tf.constant(np.array([1+1j, 2+2j, 3+3j, 4+4j]), dtype=tf.complex64)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_gradient_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    accumulator = tf.compat.v1.get_variable("accumulator_8", initializer=tf.zeros([4], dtype=tf.int64), use_resource=True, trainable=False)
    handle = tf.raw_ops.VarHandleOp(dtype=tf.int64, shape=tf.TensorShape([4]), shared_name="accumulator_handle_8", name="accumulator_handle_8", container="")
    local_step = tf.constant(8, dtype=tf.int64)
    gradient = tf.constant(np.array([1, 2, 3, 4], dtype=np.int64), dtype=tf.int64)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_gradient_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    accumulator = tf.compat.v1.get_variable("accumulator_9", initializer=tf.zeros([3], dtype=tf.float32), use_resource=True, trainable=False)
    handle = tf.raw_ops.VarHandleOp(dtype=tf.float32, shape=tf.TensorShape([3]), shared_name="accumulator_handle_9", name="accumulator_handle_9", container="")
    local_step = tf.constant(9, dtype=tf.int64)
    gradient = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32), dtype=tf.float32)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_gradient_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    accumulator = tf.compat.v1.get_variable("accumulator_10", initializer=tf.zeros([2, 2], dtype=tf.float64), use_resource=True, trainable=False)
    handle = tf.raw_ops.VarHandleOp(dtype=tf.float64, shape=tf.TensorShape([2, 2]), shared_name="accumulator_handle_10", name="accumulator_handle_10", container="")
    local_step = tf.constant(10, dtype=tf.int64)
    gradient = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.float64), dtype=tf.float64)
    input_dict = {"handle": handle, "local_step": local_step, "gradient": gradient, "name": "apply_gradient_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AccumulatorApplyGradient"] = tf_raw_ops_AccumulatorApplyGradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AccumulatorApplyGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorApplyGradient'.")

check_valid('tf.raw_ops.AccumulatorApplyGradient', generated_inputs['tf.raw_ops.AccumulatorApplyGradient'], lib="tf", suffix=0)
