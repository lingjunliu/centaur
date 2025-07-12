
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_proximal_adagrad_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([0.5, -0.5, 1.0], dtype=np.float32)
    use_locking = False
    name = "adagrad_1"

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

    # Input 2
    var = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    accum = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    lr = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.1, dtype=np.float32)
    l2 = np.array(0.1, dtype=np.float32)
    grad = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    use_locking = True
    name = "adagrad_2"

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

    # Input 3 - Int32
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([1, 2, 3], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    grad = np.array([1, -1, 2], dtype=np.int32)
    use_locking = False
    name = "adagrad_3"

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

    # Input 4 - different values
    var = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    accum = np.array([0.25, 2.25, 6.25], dtype=np.float64)
    lr = np.array(0.05, dtype=np.float64)
    l1 = np.array(0.02, dtype=np.float64)
    l2 = np.array(0.01, dtype=np.float64)
    grad = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    use_locking = True
    name = "adagrad_4"

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

    # Input 5 - negative values
    var = np.array([-0.5, -1.5, -2.5], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    use_locking = False
    name = "adagrad_5"

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

    # Input 6 - 2D array
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([[0.5, -0.5], [1.0, -1.0]], dtype=np.float32)
    use_locking = True
    name = "adagrad_6"

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

     # Input 7 - uint8
    var = np.array([1, 2, 3], dtype=np.uint8)
    accum = np.array([1, 2, 3], dtype=np.uint8)
    lr = np.array(1, dtype=np.uint8)
    l1 = np.array(0, dtype=np.uint8)
    l2 = np.array(0, dtype=np.uint8)
    grad = np.array([1, 1, 2], dtype=np.uint8)
    use_locking = False
    name = "adagrad_7"

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

    # Input 8 - complex64
    var = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    accum = np.array([0.1+0.1j, 0.2+0.2j, 0.3+0.3j], dtype=np.complex64)
    lr = np.array(0.01+0.01j, dtype=np.complex64)
    l1 = np.array(0.0+0.0j, dtype=np.complex64)
    l2 = np.array(0.0+0.0j, dtype=np.complex64)
    grad = np.array([0.5+0.5j, -0.5-0.5j, 1.0+1.0j], dtype=np.complex64)
    use_locking = False
    name = "adagrad_8"

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

    # Input 9 - l1 and l2 regularization
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.01, dtype=np.float32)
    l2 = np.array(0.01, dtype=np.float32)
    grad = np.array([0.5, -0.5, 1.0], dtype=np.float32)
    use_locking = False
    name = "adagrad_9"

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

    # Input 10 - half
    var = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    lr = np.array(0.01, dtype=np.float16)
    l1 = np.array(0.0, dtype=np.float16)
    l2 = np.array(0.0, dtype=np.float16)
    grad = np.array([0.5, -0.5, 1.0], dtype=np.float16)
    use_locking = False
    name = "adagrad_10"

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
generated_inputs["tf.raw_ops.ApplyProximalAdagrad"] = tf_raw_ops_apply_proximal_adagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyProximalAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyProximalAdagrad'.")

check_valid('tf.raw_ops.ApplyProximalAdagrad', generated_inputs['tf.raw_ops.ApplyProximalAdagrad'], lib="tf", suffix=0)
