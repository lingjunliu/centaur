
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ApplyAdagradDA_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    gradient_accumulator = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    gradient_squared_accumulator = np.array([0.01, 0.04, 0.09], dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.001, dtype=np.float32)
    l2 = np.array(0.002, dtype=np.float32)
    global_step = np.array(10, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_1"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    gradient_squared_accumulator = np.array([[0.01, 0.04], [0.09, 0.16]], dtype=np.float64)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
    lr = np.array(0.005, dtype=np.float64)
    l1 = np.array(0.0005, dtype=np.float64)
    l2 = np.array(0.001, dtype=np.float64)
    global_step = np.array(20, dtype=np.int64)
    use_locking = True
    name = "adagrad_da_2"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([1, 2, 3], dtype=np.int32)
    gradient_accumulator = np.array([0, 0, 0], dtype=np.int32)
    gradient_squared_accumulator = np.array([0, 0, 0], dtype=np.int32)
    grad = np.array([1, 1, 1], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    global_step = np.array(30, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_3"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    var = np.array([1.0], dtype=np.float32)
    gradient_accumulator = np.array([0.1], dtype=np.float32)
    gradient_squared_accumulator = np.array([0.01], dtype=np.float32)
    grad = np.array([-0.5], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.001, dtype=np.float32)
    l2 = np.array(0.002, dtype=np.float32)
    global_step = np.array(10, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_4"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    var = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    gradient_accumulator = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    gradient_squared_accumulator = np.array([[[0.01, 0.04], [0.09, 0.16]], [[0.25, 0.36], [0.49, 0.64]]], dtype=np.float32)
    grad = np.array([[[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]], dtype=np.float32)
    lr = np.array(0.001, dtype=np.float32)
    l1 = np.array(0.0001, dtype=np.float32)
    l2 = np.array(0.0002, dtype=np.float32)
    global_step = np.array(40, dtype=np.int64)
    use_locking = True
    name = "adagrad_da_5"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64 type
    var = np.array([1, 2, 3], dtype=np.int64)
    gradient_accumulator = np.array([0, 0, 0], dtype=np.int64)
    gradient_squared_accumulator = np.array([0, 0, 0], dtype=np.int64)
    grad = np.array([1, 1, 1], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    l1 = np.array(0, dtype=np.int64)
    l2 = np.array(0, dtype=np.int64)
    global_step = np.array(50, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_6"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8 type
    var = np.array([1, 2, 3], dtype=np.uint8)
    gradient_accumulator = np.array([0, 0, 0], dtype=np.uint8)
    gradient_squared_accumulator = np.array([0, 0, 0], dtype=np.uint8)
    grad = np.array([1, 1, 1], dtype=np.uint8)
    lr = np.array(1, dtype=np.uint8)
    l1 = np.array(0, dtype=np.uint8)
    l2 = np.array(0, dtype=np.uint8)
    global_step = np.array(60, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_7"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half type (float16)
    var = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    gradient_accumulator = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    gradient_squared_accumulator = np.array([0.01, 0.04, 0.09], dtype=np.float16)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float16)
    lr = np.array(0.01, dtype=np.float16)
    l1 = np.array(0.001, dtype=np.float16)
    l2 = np.array(0.002, dtype=np.float16)
    global_step = np.array(70, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_8"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex64 type
    var = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    gradient_accumulator = np.array([0+0j, 0+0j, 0+0j], dtype=np.complex64)
    gradient_squared_accumulator = np.array([0+0j, 0+0j, 0+0j], dtype=np.complex64)
    grad = np.array([1+1j, 1+1j, 1+1j], dtype=np.complex64)
    lr = np.array(0.1+0.1j, dtype=np.complex64)
    l1 = np.array(0, dtype=np.complex64)
    l2 = np.array(0, dtype=np.complex64)
    global_step = np.array(80, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_9"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    gradient_accumulator = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    gradient_squared_accumulator = np.array([0.01, 0.04, 0.09], dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.001, dtype=np.float32)
    l2 = np.array(0.002, dtype=np.float32)
    global_step = np.array(10, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_11"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))



    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyAdagradDA"] = tf_raw_ops_ApplyAdagradDA_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyAdagradDA' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdagradDA'.")

check_valid('tf.raw_ops.ApplyAdagradDA', generated_inputs['tf.raw_ops.ApplyAdagradDA'], lib="tf", suffix=0)
