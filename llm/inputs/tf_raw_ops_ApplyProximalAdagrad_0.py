
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ApplyProximalAdagrad_inputs():
    list_of_inputs = []

    # Input 1, valid
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.001, dtype=np.float32)
    l2 = np.array(0.002, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    use_locking = False
    name = "adagrad_update_1"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid, different shape
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.001, dtype=np.float32)
    l2 = np.array(0.002, dtype=np.float32)
    grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    use_locking = True
    name = "adagrad_update_2"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, int32
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([1, 2, 3], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    grad = np.array([1, 2, 3], dtype=np.int32)
    use_locking = False
    name = "adagrad_update_3"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, negative values
    var = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.001, dtype=np.float32)
    l2 = np.array(0.002, dtype=np.float32)
    grad = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    use_locking = True
    name = "adagrad_update_4"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, l1 and l2 regularization
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.1, dtype=np.float32)
    l2 = np.array(0.2, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    use_locking = False
    name = "adagrad_update_5"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, zero lr
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.0, dtype=np.float32)
    l1 = np.array(0.001, dtype=np.float32)
    l2 = np.array(0.002, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    use_locking = True
    name = "adagrad_update_6"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7, valid, int64
    var = np.array([1, 2, 3], dtype=np.int64)
    accum = np.array([1, 2, 3], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    l1 = np.array(0, dtype=np.int64)
    l2 = np.array(0, dtype=np.int64)
    grad = np.array([1, 2, 3], dtype=np.int64)
    use_locking = False
    name = "adagrad_update_7"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, float64
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    l1 = np.array(0.001, dtype=np.float64)
    l2 = np.array(0.002, dtype=np.float64)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    use_locking = True
    name = "adagrad_update_8"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, 3D tensor
    var = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    accum = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.001, dtype=np.float32)
    l2 = np.array(0.002, dtype=np.float32)
    grad = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    use_locking = False
    name = "adagrad_update_9"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10, valid, float16
    var = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    lr = np.array(0.01, dtype=np.float16)
    l1 = np.array(0.001, dtype=np.float16)
    l2 = np.array(0.002, dtype=np.float16)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    use_locking = False
    name = "adagrad_update_10"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyProximalAdagrad"] = tf_raw_ops_ApplyProximalAdagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyProximalAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyProximalAdagrad'.")

check_valid('tf.raw_ops.ApplyProximalAdagrad', generated_inputs['tf.raw_ops.ApplyProximalAdagrad'], lib="tf", suffix=0)
