
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_momentum_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_1"

    var_tensor = tf.Variable(var)
    accum_tensor = tf.Variable(accum)

    input_dict = {
        "var": var_tensor,
        "accum": accum_tensor,
        "lr": tf.convert_to_tensor(lr),
        "grad": grad,
        "momentum": tf.convert_to_tensor(momentum),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    accum = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    lr = np.array(0.001, dtype=np.float64)
    grad = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    momentum = np.array(0.99, dtype=np.float64)
    use_locking = True
    use_nesterov = True
    name = "momentum_update_2"

    var_tensor = tf.Variable(var)
    accum_tensor = tf.Variable(accum)

    input_dict = {
        "var": var_tensor,
        "accum": accum_tensor,
        "lr": tf.convert_to_tensor(lr),
        "grad": grad,
        "momentum": tf.convert_to_tensor(momentum),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([0, 0, 0], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    grad = np.array([1, 2, 3], dtype=np.int32)
    momentum = np.array(1, dtype=np.int32)
    use_locking = False
    use_nesterov = True
    name = "momentum_update_3"

    var_tensor = tf.Variable(var)
    accum_tensor = tf.Variable(accum)

    input_dict = {
        "var": var_tensor,
        "accum": accum_tensor,
        "lr": tf.convert_to_tensor(lr),
        "grad": grad,
        "momentum": tf.convert_to_tensor(momentum),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = np.array([1, 2, 3], dtype=np.int64)
    accum = np.array([0, 0, 0], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    grad = np.array([1, 2, 3], dtype=np.int64)
    momentum = np.array(1, dtype=np.int64)
    use_locking = True
    use_nesterov = False
    name = "momentum_update_4"

    var_tensor = tf.Variable(var)
    accum_tensor = tf.Variable(accum)

    input_dict = {
        "var": var_tensor,
        "accum": accum_tensor,
        "lr": tf.convert_to_tensor(lr),
        "grad": grad,
        "momentum": tf.convert_to_tensor(momentum),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_5"

    var_tensor = tf.Variable(var)
    accum_tensor = tf.Variable(accum)

    input_dict = {
        "var": var_tensor,
        "accum": accum_tensor,
        "lr": tf.convert_to_tensor(lr),
        "grad": grad,
        "momentum": tf.convert_to_tensor(momentum),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    var = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    accum = np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_6"

    var_tensor = tf.Variable(var)
    accum_tensor = tf.Variable(accum)

    input_dict = {
        "var": var_tensor,
        "accum": accum_tensor,
        "lr": tf.convert_to_tensor(lr),
        "grad": grad,
        "momentum": tf.convert_to_tensor(momentum),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.0], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.1], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_7"

    var_tensor = tf.Variable(var)
    accum_tensor = tf.Variable(accum)

    input_dict = {
        "var": var_tensor,
        "accum": accum_tensor,
        "lr": tf.convert_to_tensor(lr),
        "grad": grad,
        "momentum": tf.convert_to_tensor(momentum),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16
    var = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    accum = np.array([0.0, 0.0, 0.0], dtype=np.float16)
    lr = np.array(0.01, dtype=np.float16)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    momentum = np.array(0.9, dtype=np.float16)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_8"

    var_tensor = tf.Variable(var)
    accum_tensor = tf.Variable(accum)

    input_dict = {
        "var": var_tensor,
        "accum": accum_tensor,
        "lr": tf.convert_to_tensor(lr),
        "grad": grad,
        "momentum": tf.convert_to_tensor(momentum),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    var = np.array([1, 2, 3], dtype=np.uint8)
    accum = np.array([0, 0, 0], dtype=np.uint8)
    lr = np.array(1, dtype=np.uint8)
    grad = np.array([1, 2, 3], dtype=np.uint8)
    momentum = np.array(1, dtype=np.uint8)
    use_locking = True
    use_nesterov = False
    name = "momentum_update_9"

    var_tensor = tf.Variable(var)
    accum_tensor = tf.Variable(accum)

    input_dict = {
        "var": var_tensor,
        "accum": accum_tensor,
        "lr": tf.convert_to_tensor(lr),
        "grad": grad,
        "momentum": tf.convert_to_tensor(momentum),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    var = np.array([1, 2, 3], dtype=np.uint64)
    accum = np.array([0, 0, 0], dtype=np.uint64)
    lr = np.array(1, dtype=np.uint64)
    grad = np.array([1, 2, 3], dtype=np.uint64)
    momentum = np.array(1, dtype=np.uint64)
    use_locking = True
    use_nesterov = False
    name = "momentum_update_10"

    var_tensor = tf.Variable(var)
    accum_tensor = tf.Variable(accum)

    input_dict = {
        "var": var_tensor,
        "accum": accum_tensor,
        "lr": tf.convert_to_tensor(lr),
        "grad": grad,
        "momentum": tf.convert_to_tensor(momentum),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyMomentum"] = tf_raw_ops_apply_momentum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyMomentum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyMomentum'.")

check_valid('tf.raw_ops.ApplyMomentum', generated_inputs['tf.raw_ops.ApplyMomentum'], lib="tf", suffix=0)
