
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseApplyFtrl_inputs():
    list_of_inputs = []

    # Input 1: Basic test case with float32
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    linear = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    grad = np.array([[0.2, 0.3]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "sparse_apply_ftrl_1"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Using float64 and different parameters
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    linear = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
    grad = np.array([[0.2, 0.3]], dtype=np.float64)
    indices = np.array([0], dtype=np.int64)
    lr = np.array(0.01, dtype=np.float64)
    l1 = np.array(0.1, dtype=np.float64)
    l2 = np.array(0.01, dtype=np.float64)
    lr_power = np.array(-0.5, dtype=np.float64)
    use_locking = True
    multiply_linear_by_lr = True
    name = "sparse_apply_ftrl_2"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple indices
    var = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    linear = np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
    grad = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "sparse_apply_ftrl_3"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different lr_power
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    linear = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    grad = np.array([[0.2, 0.3]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(0.0, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "sparse_apply_ftrl_5"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: l1 regularization
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    linear = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    grad = np.array([[0.2, 0.3]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.5, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "sparse_apply_ftrl_6"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: l2 regularization
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    linear = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    grad = np.array([[0.2, 0.3]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.1, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "sparse_apply_ftrl_7"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: multiply_linear_by_lr = True
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    linear = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    grad = np.array([[0.2, 0.3]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = True
    name = "sparse_apply_ftrl_8"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger indices, var with more rows
    var = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    linear = np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2]], dtype=np.float32)
    grad = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]], dtype=np.float32)
    indices = np.array([0, 2, 3], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "sparse_apply_ftrl_10"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int32 var,accum,linear,grad
    var = np.array([[1, 2], [3, 4]], dtype=np.int32)
    accum = np.array([[1, 2], [3, 4]], dtype=np.int32)
    linear = np.array([[1, 2], [3, 4]], dtype=np.int32)
    grad = np.array([[1, 2]], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    lr_power = np.array(0, dtype=np.int32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "sparse_apply_ftrl_9"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64 indices
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    linear = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    grad = np.array([[0.2, 0.3]], dtype=np.float32)
    indices = np.array([0], dtype=np.int64)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "sparse_apply_ftrl_11"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseApplyFtrl"] = tf_raw_ops_SparseApplyFtrl_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseApplyFtrl' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyFtrl'.")

check_valid('tf.raw_ops.SparseApplyFtrl', generated_inputs['tf.raw_ops.SparseApplyFtrl'], lib="tf", suffix=0)
