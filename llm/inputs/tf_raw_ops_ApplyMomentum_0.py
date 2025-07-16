
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_momentum_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    accum = tf.Variable(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "momentum": momentum, "use_locking": False, "use_nesterov": False, "name": "momentum_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 with Nesterov
    var = tf.Variable(np.array([1.0, 2.0], dtype=np.float64))
    accum = tf.Variable(np.array([0.5, 0.5], dtype=np.float64))
    lr = np.array(0.001, dtype=np.float64)
    grad = np.array([0.05, 0.1], dtype=np.float64)
    momentum = np.array(0.95, dtype=np.float64)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "momentum": momentum, "use_locking": True, "use_nesterov": True, "name": "momentum_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32
    var = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    accum = tf.Variable(np.array([0, 0, 0], dtype=np.int32))
    lr = np.array(1, dtype=np.int32)
    grad = np.array([1, 2, 3], dtype=np.int32)
    momentum = np.array(0, dtype=np.int32)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "momentum": momentum, "use_locking": False, "use_nesterov": False, "name": "momentum_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64
    var = tf.Variable(np.array([1+1j, 2+2j], dtype=np.complex64))
    accum = tf.Variable(np.array([0+0j, 0+0j], dtype=np.complex64))
    lr = np.array(0.1+0j, dtype=np.complex64)
    grad = np.array([0.1+0.1j, 0.2+0.2j], dtype=np.complex64)
    momentum = np.array(0.9+0j, dtype=np.complex64)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "momentum": momentum, "use_locking": False, "use_nesterov": False, "name": "momentum_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    var = tf.Variable(np.array([-1.0, -2.0], dtype=np.float32))
    accum = tf.Variable(np.array([0.0, 0.0], dtype=np.float32))
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([-0.1, -0.2], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "momentum": momentum, "use_locking": False, "use_nesterov": False, "name": "momentum_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional array
    var = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    accum = tf.Variable(np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "momentum": momentum, "use_locking": False, "use_nesterov": False, "name": "momentum_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: half
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float16))
    accum = tf.Variable(np.array([0.0, 0.0, 0.0], dtype=np.float16))
    lr = np.array(0.01, dtype=np.float16)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    momentum = np.array(0.9, dtype=np.float16)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "momentum": momentum, "use_locking": False, "use_nesterov": False, "name": "momentum_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint32
    var = tf.Variable(np.array([1, 2, 3], dtype=np.uint32))
    accum = tf.Variable(np.array([0, 0, 0], dtype=np.uint32))
    lr = np.array(1, dtype=np.uint32)
    grad = np.array([1, 2, 3], dtype=np.uint32)
    momentum = np.array(0, dtype=np.uint32)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "momentum": momentum, "use_locking": False, "use_nesterov": False, "name": "momentum_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint64, use_locking=True, use_nesterov=True
    var = tf.Variable(np.array([1, 2], dtype=np.uint64))
    accum = tf.Variable(np.array([0, 0], dtype=np.uint64))
    lr = np.array(1, dtype=np.uint64)
    grad = np.array([1, 2], dtype=np.uint64)
    momentum = np.array(0, dtype=np.uint64)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "momentum": momentum, "use_locking": True, "use_nesterov": True, "name": "momentum_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64, use_locking=True, use_nesterov=True
    var = tf.Variable(np.array([1, 2], dtype=np.int64))
    accum = tf.Variable(np.array([0, 0], dtype=np.int64))
    lr = np.array(1, dtype=np.int64)
    grad = np.array([1, 2], dtype=np.int64)
    momentum = np.array(0, dtype=np.int64)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "momentum": momentum, "use_locking": True, "use_nesterov": True, "name": "momentum_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyMomentum"] = tf_raw_ops_apply_momentum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyMomentum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyMomentum'.")

check_valid('tf.raw_ops.ApplyMomentum', generated_inputs['tf.raw_ops.ApplyMomentum'], lib="tf", suffix=0)
