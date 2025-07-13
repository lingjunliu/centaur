
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_adagrad_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 example
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    use_locking = False
    update_slots = True
    name = "adagrad_1"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 example with different values
    var = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    grad = np.array([-0.1, 0.2, -0.3], dtype=np.float64)
    use_locking = True
    update_slots = False
    name = "adagrad_2"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32 example
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([1, 2, 3], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    grad = np.array([1, 2, 3], dtype=np.int32)
    use_locking = False
    update_slots = True
    name = "adagrad_3"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger values and negative learning rate (valid for int types)
    var = np.array([-10, 20, -30], dtype=np.int32)
    accum = np.array([5, 5, 5], dtype=np.int32)
    lr = np.array(-2, dtype=np.int32)
    grad = np.array([5, -5, 5], dtype=np.int32)
    use_locking = True
    update_slots = False
    name = "adagrad_4"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 example
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    use_locking = False
    update_slots = True
    name = "adagrad_5"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int32 example
    var = np.array([[1, 2], [3, 4]], dtype=np.int32)
    accum = np.array([[1, 2], [3, 4]], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    grad = np.array([[1, 2], [3, 4]], dtype=np.int32)
    use_locking = True
    update_slots = False
    name = "adagrad_6"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: float32 with zeros
    var = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    use_locking = False
    update_slots = True
    name = "adagrad_7"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64 example
    var = np.array([1, 2, 3], dtype=np.int64)
    accum = np.array([1, 2, 3], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    grad = np.array([1, 2, 3], dtype=np.int64)
    use_locking = False
    update_slots = True
    name = "adagrad_8"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex64
    var = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    accum = np.array([0.1+0.1j, 0.2+0.2j, 0.3+0.3j], dtype=np.complex64)
    lr = np.array(0.01+0.01j, dtype=np.complex64)
    grad = np.array([0.1+0.1j, 0.2+0.2j, 0.3+0.3j], dtype=np.complex64)
    use_locking = True
    update_slots = False
    name = "adagrad_9"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32).astype(np.float16)
    var = tf.constant(var)
    var = tf.cast(var, tf.bfloat16).numpy()
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32).astype(np.float16)
    accum = tf.constant(accum)
    accum = tf.cast(accum, tf.bfloat16).numpy()
    lr = np.array(0.01, dtype=np.float32).astype(np.float16)
    lr = tf.constant(lr)
    lr = tf.cast(lr, tf.bfloat16).numpy()
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32).astype(np.float16)
    grad = tf.constant(grad)
    grad = tf.cast(grad, tf.bfloat16).numpy()
    use_locking = False
    update_slots = True
    name = "adagrad_10"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyAdagrad"] = tf_raw_ops_apply_adagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdagrad'.")

check_valid('tf.raw_ops.ApplyAdagrad', generated_inputs['tf.raw_ops.ApplyAdagrad'], lib="tf", suffix=0)
