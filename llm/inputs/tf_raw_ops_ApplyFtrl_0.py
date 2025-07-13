
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_ftrl_inputs():
    list_of_inputs = []

    # Input 1: float32
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    linear = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    grad = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    input_dict = {"var": var, "accum": accum, "linear": linear, "grad": grad, "lr": lr, "l1": l1, "l2": l2, "lr_power": lr_power, "use_locking": False, "multiply_linear_by_lr": False, "name": "ftrl_float32_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    linear = np.array([0.4, 0.5, 0.6], dtype=np.float64)
    grad = np.array([0.7, 0.8, 0.9], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    l1 = np.array(0.0, dtype=np.float64)
    l2 = np.array(0.0, dtype=np.float64)
    lr_power = np.array(-0.5, dtype=np.float64)
    input_dict = {"var": var, "accum": accum, "linear": linear, "grad": grad, "lr": lr, "l1": l1, "l2": l2, "lr_power": lr_power, "use_locking": True, "multiply_linear_by_lr": True, "name": "ftrl_float64_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([0, 0, 0], dtype=np.int32)
    linear = np.array([0, 0, 0], dtype=np.int32)
    grad = np.array([1, 1, 1], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    lr_power = np.array(-1, dtype=np.int32)
    input_dict = {"var": var, "accum": accum, "linear": linear, "grad": grad, "lr": lr, "l1": l1, "l2": l2, "lr_power": lr_power, "use_locking": False, "multiply_linear_by_lr": False, "name": "ftrl_int32_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyFtrl"] = tf_raw_ops_apply_ftrl_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyFtrl' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyFtrl'.")

check_valid('tf.raw_ops.ApplyFtrl', generated_inputs['tf.raw_ops.ApplyFtrl'], lib="tf", suffix=0)
