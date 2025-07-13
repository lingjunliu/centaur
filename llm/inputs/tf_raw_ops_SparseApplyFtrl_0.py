
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparse_apply_ftrl_inputs():
    list_of_inputs = []

    # Input 1
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
        "name": "ftrl_op_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
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
        "name": "ftrl_op_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([1, 2, 3, 4], dtype=np.int32)
    accum = np.array([5, 6, 7, 8], dtype=np.int32)
    linear = np.array([9, 10, 11, 12], dtype=np.int32)
    grad = np.array([13, 14], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    lr_power = np.array(-1, dtype=np.int32)
    use_locking = False
    multiply_linear_by_lr = False

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
        "name": "ftrl_op_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    linear = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    grad = np.array([0.2, 0.3], dtype=np.float32)
    indices = np.array([0,1], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False

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
        "name": "ftrl_op_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    linear = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    grad = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
    indices = np.array([0,1], dtype=np.int64)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False

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
        "name": "ftrl_op_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    linear = np.array([0.5], dtype=np.float32)
    grad = np.array([0.2], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False

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
        "name": "ftrl_op_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = np.array([1.0, 2.0], dtype=np.float32)
    accum = np.array([0.1, 0.2], dtype=np.float32)
    linear = np.array([0.5, 0.6], dtype=np.float32)
    grad = np.array([0.2, 0.3], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int64)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False

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
        "name": "ftrl_op_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    linear = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    grad = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int64)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False

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
        "name": "ftrl_op_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    linear = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    grad = np.array([0.2, 0.3], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int64)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False

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
        "name": "ftrl_op_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    linear = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    grad = np.array([[0.2, 0.3]], dtype=np.float32)
    indices = np.array([1], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False

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
        "name": "ftrl_op_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    linear = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    grad = np.array([[0.2, 0.3]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False

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
        "name": "ftrl_op_11"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 - Test with int64 grad
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    linear = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    grad = np.array([0.2, 0.3], dtype=np.float32) #Corrected type
    indices = np.array([0, 1], dtype=np.int64)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
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
        "name": "ftrl_op_12"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    #Input 13 - Test with different var type
    var = np.array([[1, 2], [3, 4]], dtype=np.int32)
    accum = np.array([[0, 0], [0, 0]], dtype=np.int32)
    linear = np.array([[0, 0], [0, 0]], dtype=np.int32)
    grad = np.array([[1, 1]], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    lr_power = np.array(-1, dtype=np.int32)
    use_locking = False
    multiply_linear_by_lr = False
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
        "name": "ftrl_op_13"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14
    var = np.array([1.0, 2.0], dtype=np.float32)
    accum = np.array([0.1, 0.2], dtype=np.float32)
    linear = np.array([0.5, 0.6], dtype=np.float32)
    grad = np.array([0.2, 0.3], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False

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
        "name": "ftrl_op_14"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseApplyFtrl"] = tf_raw_ops_sparse_apply_ftrl_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseApplyFtrl' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyFtrl'.")

check_valid('tf.raw_ops.SparseApplyFtrl', generated_inputs['tf.raw_ops.SparseApplyFtrl'], lib="tf", suffix=0)
