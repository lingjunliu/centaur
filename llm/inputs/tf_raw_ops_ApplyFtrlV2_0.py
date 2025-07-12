
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_ftrl_v2_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    linear = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.1, dtype=np.float32)
    l2 = np.array(0.01, dtype=np.float32)
    l2_shrinkage = np.array(0.001, dtype=np.float32)
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
        "name": "ftrl_basic"
    }
    list_of_inputs.append(input_dict)

    # Input 2: float64, different values
    var = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    accum = np.array([0.5, 0.6, 0.7], dtype=np.float64)
    linear = np.array([0.2, 0.3, 0.4], dtype=np.float64)
    grad = np.array([-0.1, -0.2, -0.3], dtype=np.float64)
    lr = np.array(0.005, dtype=np.float64)
    l1 = np.array(0.05, dtype=np.float64)
    l2 = np.array(0.005, dtype=np.float64)
    l2_shrinkage = np.array(0.0005, dtype=np.float64)
    lr_power = np.array(-0.25, dtype=np.float64)

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
        "multiply_linear_by_lr": True,
        "name": "ftrl_float64"
    }
    list_of_inputs.append(input_dict)

    # Input 3: int32
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([1, 2, 3], dtype=np.int32)
    linear = np.array([0, 0, 0], dtype=np.int32)
    grad = np.array([1, 1, 1], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(1, dtype=np.int32)
    l2 = np.array(1, dtype=np.int32)
    l2_shrinkage = np.array(1, dtype=np.int32)
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
        "name": "ftrl_int32"
    }
    list_of_inputs.append(input_dict)

   # Input 4: Different shapes
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    linear = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.1, dtype=np.float32)
    l2 = np.array(0.01, dtype=np.float32)
    l2_shrinkage = np.array(0.001, dtype=np.float32)
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
        "name": "ftrl_shape"
    }
    list_of_inputs.append(input_dict)

    # Input 5: all zeros
    var = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    linear = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    grad = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.1, dtype=np.float32)
    l2 = np.array(0.01, dtype=np.float32)
    l2_shrinkage = np.array(0.001, dtype=np.float32)
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
        "name": "ftrl_zeros"
    }
    list_of_inputs.append(input_dict)

    # Input 6: large values
    var = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
    accum = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    linear = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    grad = np.array([50.0, 60.0, 70.0], dtype=np.float32)
    lr = np.array(0.1, dtype=np.float32)
    l1 = np.array(10.0, dtype=np.float32)
    l2 = np.array(1.0, dtype=np.float32)
    l2_shrinkage = np.array(0.1, dtype=np.float32)
    lr_power = np.array(-0.1, dtype=np.float32)

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
        "name": "ftrl_large"
    }
    list_of_inputs.append(input_dict)

    # Input 7: lr_power close to 0
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    linear = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.1, dtype=np.float32)
    l2 = np.array(0.01, dtype=np.float32)
    l2_shrinkage = np.array(0.001, dtype=np.float32)
    lr_power = np.array(-0.001, dtype=np.float32)

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
        "name": "ftrl_lr_power_close_zero"
    }
    list_of_inputs.append(input_dict)

    # Input 8: lr=1, l1=0, l2=0, l2_shrinkage=0, lr_power=0
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    linear = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    lr = np.array(1.0, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    l2_shrinkage = np.array(0.0, dtype=np.float32)
    lr_power = np.array(0.0, dtype=np.float32)

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
        "name": "ftrl_special_params"
    }
    list_of_inputs.append(input_dict)

    # Input 9: int64
    var = np.array([1, 2, 3], dtype=np.int64)
    accum = np.array([1, 2, 3], dtype=np.int64)
    linear = np.array([0, 0, 0], dtype=np.int64)
    grad = np.array([1, 1, 1], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    l1 = np.array(1, dtype=np.int64)
    l2 = np.array(1, dtype=np.int64)
    l2_shrinkage = np.array(1, dtype=np.int64)
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
        "name": "ftrl_int64"
    }
    list_of_inputs.append(input_dict)

    # Input 10: multiple dimensions
    var = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    accum = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    linear = np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    grad = np.array([[[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.1, dtype=np.float32)
    l2 = np.array(0.01, dtype=np.float32)
    l2_shrinkage = np.array(0.001, dtype=np.float32)
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
        "name": "ftrl_multi_dim"
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_apply_ftrl_v2_inputs()
generated_inputs["tf.raw_ops.ApplyFtrlV2"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyFtrlV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyFtrlV2'.")

check_valid('tf.raw_ops.ApplyFtrlV2', generated_inputs['tf.raw_ops.ApplyFtrlV2'], lib="tf", suffix=0)
