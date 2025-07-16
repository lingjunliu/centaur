
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_adagrad_inputs():
    list_of_inputs = []

    def create_input_dict(var, accum, lr, grad, use_locking, update_slots, name):
        return {
            "var": var,
            "accum": accum,
            "lr": lr,
            "grad": grad,
            "use_locking": use_locking,
            "update_slots": update_slots,
            "name": name
        }

    # Input 1: Basic float32 example
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    accum = tf.Variable(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    lr = tf.constant(0.01, dtype=np.float32)
    grad = tf.constant([0.1, 0.2, 0.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(create_input_dict(var, accum, lr, grad, False, True, "adagrad_1")))

    # Input 2: Float64 example
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    accum = tf.Variable(np.array([0.1, 0.2, 0.3], dtype=np.float64))
    lr = tf.constant(0.01, dtype=np.float64)
    grad = tf.constant([0.1, 0.2, 0.3], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy(create_input_dict(var, accum, lr, grad, True, False, "adagrad_2")))

    # Input 3: Int32 example
    var = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    accum = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    lr = tf.constant(1, dtype=np.int32)
    grad = tf.constant([1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy(create_input_dict(var, accum, lr, grad, False, True, "adagrad_3")))

    # Input 4: Negative values
    var = tf.Variable(np.array([-1.0, -2.0, -3.0], dtype=np.float32))
    accum = tf.Variable(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    lr = tf.constant(0.01, dtype=np.float32)
    grad = tf.constant([-0.1, -0.2, -0.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(create_input_dict(var, accum, lr, grad, True, False, "adagrad_4")))

    # Input 5: Multi-dimensional array
    var = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    accum = tf.Variable(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    lr = tf.constant(0.01, dtype=np.float32)
    grad = tf.constant([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(create_input_dict(var, accum, lr, grad, False, True, "adagrad_5")))

    # Input 6: Int64
    var = tf.Variable(np.array([1, 2, 3], dtype=np.int64))
    accum = tf.Variable(np.array([1, 2, 3], dtype=np.int64))
    lr = tf.constant(1, dtype=np.int64)
    grad = tf.constant([1, 2, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy(create_input_dict(var, accum, lr, grad, True, False, "adagrad_6")))
    
    # Input 7: uint8
    var = tf.Variable(np.array([1, 2, 3], dtype=np.uint8))
    accum = tf.Variable(np.array([1, 2, 3], dtype=np.uint8))
    lr = tf.constant(1, dtype=np.uint8)
    grad = tf.constant([1, 2, 3], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(var, accum, lr, grad, False, True, "adagrad_7")))

    # Input 8: int16
    var = tf.Variable(np.array([1, 2, 3], dtype=np.int16))
    accum = tf.Variable(np.array([1, 2, 3], dtype=np.int16))
    lr = tf.constant(1, dtype=np.int16)
    grad = tf.constant([1, 2, 3], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy(create_input_dict(var, accum, lr, grad, True, False, "adagrad_8")))

    # Input 9: uint16
    var = tf.Variable(np.array([1, 2, 3], dtype=np.uint16))
    accum = tf.Variable(np.array([1, 2, 3], dtype=np.uint16))
    lr = tf.constant(1, dtype=np.uint16)
    grad = tf.constant([1, 2, 3], dtype=np.uint16)
    list_of_inputs.append(copy.deepcopy(create_input_dict(var, accum, lr, grad, False, True, "adagrad_9")))

    # Input 10: int8
    var = tf.Variable(np.array([1, 2, 3], dtype=np.int8))
    accum = tf.Variable(np.array([1, 2, 3], dtype=np.int8))
    lr = tf.constant(1, dtype=np.int8)
    grad = tf.constant([1, 2, 3], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(var, accum, lr, grad, True, False, "adagrad_10")))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyAdagrad"] = tf_raw_ops_apply_adagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdagrad'.")

check_valid('tf.raw_ops.ApplyAdagrad', generated_inputs['tf.raw_ops.ApplyAdagrad'], lib="tf", suffix=0)
