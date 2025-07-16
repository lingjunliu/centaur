
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_ftrl_v2_inputs():
    list_of_inputs = []

    # Input 1: float32, basic case
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    accum = tf.Variable(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    linear = tf.Variable(np.array([0.4, 0.5, 0.6], dtype=np.float32))
    grad = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    l2_shrinkage = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": False,
        "multiply_linear_by_lr": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, with l1 and l2 regularization
    var = tf.Variable(np.array([1.0, 2.0], dtype=np.float64))
    accum = tf.Variable(np.array([0.1, 0.2], dtype=np.float64))
    linear = tf.Variable(np.array([0.4, 0.5], dtype=np.float64))
    grad = np.array([-0.7, 0.8], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    l1 = np.array(0.1, dtype=np.float64)
    l2 = np.array(0.01, dtype=np.float64)
    l2_shrinkage = np.array(0.01, dtype=np.float64)
    lr_power = np.array(-0.5, dtype=np.float64)

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": True,
        "multiply_linear_by_lr": False,
        "name": "ftrl_op"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, with negative lr_power
    var = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    accum = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    linear = tf.Variable(np.array([4, 5, 6], dtype=np.int32))
    grad = np.array([7, 8, 9], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    l2_shrinkage = np.array(0, dtype=np.int32)
    lr_power = np.array(-1, dtype=np.int32)

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": False,
        "multiply_linear_by_lr": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4: uint8, with larger values
    var = tf.Variable(np.array([200, 210, 220], dtype=np.uint8))
    accum = tf.Variable(np.array([10, 20, 30], dtype=np.uint8))
    linear = tf.Variable(np.array([40, 50, 60], dtype=np.uint8))
    grad = np.array([70, 80, 90], dtype=np.uint8)
    lr = np.array(1, dtype=np.uint8)
    l1 = np.array(0, dtype=np.uint8)
    l2 = np.array(0, dtype=np.uint8)
    l2_shrinkage = np.array(0, dtype=np.uint8)
    lr_power = np.array(-1, dtype=np.uint8)

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": False,
        "multiply_linear_by_lr": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int16, with negative gradient
    var = tf.Variable(np.array([100, 200], dtype=np.int16))
    accum = tf.Variable(np.array([10, 20], dtype=np.int16))
    linear = tf.Variable(np.array([40, 50], dtype=np.int16))
    grad = np.array([-70, 80], dtype=np.int16)
    lr = np.array(1, dtype=np.int16)
    l1 = np.array(0, dtype=np.int16)
    l2 = np.array(0, dtype=np.int16)
    l2_shrinkage = np.array(0, dtype=np.int16)
    lr_power = np.array(-1, dtype=np.int16)

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": False,
        "multiply_linear_by_lr": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int8, with l2 regularization
    var = tf.Variable(np.array([10, 20], dtype=np.int8))
    accum = tf.Variable(np.array([1, 2], dtype=np.int8))
    linear = tf.Variable(np.array([4, 5], dtype=np.int8))
    grad = np.array([7, 8], dtype=np.int8)
    lr = np.array(1, dtype=np.int8)
    l1 = np.array(0, dtype=np.int8)
    l2 = np.array(1, dtype=np.int8)
    l2_shrinkage = np.array(1, dtype=np.int8)
    lr_power = np.array(-1, dtype=np.int8)

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": False,
        "multiply_linear_by_lr": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64, with complex gradient
    var = tf.Variable(np.array([1 + 1j, 2 + 2j], dtype=np.complex64))
    accum = tf.Variable(np.array([0.1 + 0.1j, 0.2 + 0.2j], dtype=np.complex64))
    linear = tf.Variable(np.array([0.4 + 0.4j, 0.5 + 0.5j], dtype=np.complex64))
    grad = np.array([0.7 + 0.7j, 0.8 + 0.8j], dtype=np.complex64)
    lr = np.array(0.01 + 0.01j, dtype=np.complex64)
    l1 = np.array(0, dtype=np.complex64)
    l2 = np.array(0, dtype=np.complex64)
    l2_shrinkage = np.array(0, dtype=np.complex64)
    lr_power = np.array(-0.5 + 0j, dtype=np.complex64)

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": False,
        "multiply_linear_by_lr": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64, with larger learning rate
    var = tf.Variable(np.array([1000, 2000], dtype=np.int64))
    accum = tf.Variable(np.array([100, 200], dtype=np.int64))
    linear = tf.Variable(np.array([400, 500], dtype=np.int64))
    grad = np.array([700, 800], dtype=np.int64)
    lr = np.array(10, dtype=np.int64)
    l1 = np.array(0, dtype=np.int64)
    l2 = np.array(0, dtype=np.int64)
    l2_shrinkage = np.array(0, dtype=np.int64)
    lr_power = np.array(-1, dtype=np.int64)

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": False,
        "multiply_linear_by_lr": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half, 2D array
    var = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16))
    accum = tf.Variable(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float16))
    linear = tf.Variable(np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float16))
    grad = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float16)
    lr = np.array(0.01, dtype=np.float16)
    l1 = np.array(0.0, dtype=np.float16)
    l2 = np.array(0.0, dtype=np.float16)
    l2_shrinkage = np.array(0.0, dtype=np.float16)
    lr_power = np.array(-0.5, dtype=np.float16)

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": False,
        "multiply_linear_by_lr": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float16, different values. Replacing bfloat16 as it caused error.
    var = tf.Variable(np.array([1.1, 2.2, 3.3], dtype=np.float16))
    accum = tf.Variable(np.array([0.11, 0.22, 0.33], dtype=np.float16))
    linear = tf.Variable(np.array([0.44, 0.55, 0.66], dtype=np.float16))
    grad = np.array([0.77, 0.88, 0.99], dtype=np.float16)
    lr = np.array(0.01, dtype=np.float16)
    l1 = np.array(0.0, dtype=np.float16)
    l2 = np.array(0.0, dtype=np.float16)
    l2_shrinkage = np.array(0.0, dtype=np.float16)
    lr_power = np.array(-0.5, dtype=np.float16)

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": False,
        "multiply_linear_by_lr": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyFtrlV2"] = tf_raw_ops_apply_ftrl_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyFtrlV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyFtrlV2'.")

check_valid('tf.raw_ops.ApplyFtrlV2', generated_inputs['tf.raw_ops.ApplyFtrlV2'], lib="tf", suffix=0)
