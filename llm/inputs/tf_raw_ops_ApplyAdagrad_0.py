
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_adagrad_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": False, "update_slots": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.1, dtype=np.float32)
    grad = np.array([-0.5, -0.6, -0.7], dtype=np.float32)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": True, "update_slots": False, "name": "test_adagrad"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": False, "update_slots": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([1, 2, 3], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    grad = np.array([1, 1, 1], dtype=np.int32)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": True, "update_slots": False, "name": "test_adagrad_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.5], dtype=np.float32)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": False, "update_slots": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(-0.01, dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": False, "update_slots": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([-0.5, -0.6, -0.7], dtype=np.float32)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": False, "update_slots": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([[-0.5, 0.6], [0.7, -0.8]], dtype=np.float32)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": False, "update_slots": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float64)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": False, "update_slots": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": False, "update_slots": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    var = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": False, "update_slots": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
