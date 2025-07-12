
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ApplyAdam_inputs():
    list_of_inputs = []

    # Input 1: float32, basic
    var = np.array([1.0, 2.0], dtype=np.float32)
    m = np.array([0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 0.0], dtype=np.float32)
    beta1_power = np.array(0.9, dtype=np.float32)
    beta2_power = np.array(0.999, dtype=np.float32)
    lr = np.array(0.001, dtype=np.float32)
    beta1 = np.array(0.9, dtype=np.float32)
    beta2 = np.array(0.999, dtype=np.float32)
    epsilon = np.array(1e-07, dtype=np.float32)
    grad = np.array([0.1, 0.2], dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "adam_1"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": tf.constant(beta1_power),
        "beta2_power": tf.constant(beta2_power),
        "lr": tf.constant(lr),
        "beta1": tf.constant(beta1),
        "beta2": tf.constant(beta2),
        "epsilon": tf.constant(epsilon),
        "grad": tf.constant(grad),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, with locking
    var = np.array([1.0, 2.0], dtype=np.float64)
    m = np.array([0.0, 0.0], dtype=np.float64)
    v = np.array([0.0, 0.0], dtype=np.float64)
    beta1_power = np.array(0.9, dtype=np.float64)
    beta2_power = np.array(0.999, dtype=np.float64)
    lr = np.array(0.001, dtype=np.float64)
    beta1 = np.array(0.9, dtype=np.float64)
    beta2 = np.array(0.999, dtype=np.float64)
    epsilon = np.array(1e-08, dtype=np.float64)
    grad = np.array([0.1, 0.2], dtype=np.float64)
    use_locking = True
    use_nesterov = False
    name = "adam_2"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": tf.constant(beta1_power),
        "beta2_power": tf.constant(beta2_power),
        "lr": tf.constant(lr),
        "beta1": tf.constant(beta1),
        "beta2": tf.constant(beta2),
        "epsilon": tf.constant(epsilon),
        "grad": tf.constant(grad),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, with nesterov
    var = np.array([1, 2], dtype=np.int32)
    m = np.array([0, 0], dtype=np.int32)
    v = np.array([0, 0], dtype=np.int32)
    beta1_power = np.array(0.9, dtype=np.int32)
    beta2_power = np.array(0.999, dtype=np.int32)
    lr = np.array(0.001, dtype=np.int32)
    beta1 = np.array(0.9, dtype=np.int32)
    beta2 = np.array(0.999, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array([1, 2], dtype=np.int32)
    use_locking = False
    use_nesterov = True
    name = "adam_3"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": tf.constant(beta1_power),
        "beta2_power": tf.constant(beta2_power),
        "lr": tf.constant(lr),
        "beta1": tf.constant(beta1),
        "beta2": tf.constant(beta2),
        "epsilon": tf.constant(epsilon),
        "grad": tf.constant(grad),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4: float32, different learning rate
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    m = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    beta1_power = np.array(0.8, dtype=np.float32)
    beta2_power = np.array(0.99, dtype=np.float32)
    lr = np.array(0.1, dtype=np.float32)
    beta1 = np.array(0.8, dtype=np.float32)
    beta2 = np.array(0.99, dtype=np.float32)
    epsilon = np.array(1e-07, dtype=np.float32)
    grad = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "adam_4"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": tf.constant(beta1_power),
        "beta2_power": tf.constant(beta2_power),
        "lr": tf.constant(lr),
        "beta1": tf.constant(beta1),
        "beta2": tf.constant(beta2),
        "epsilon": tf.constant(epsilon),
        "grad": tf.constant(grad),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, different beta values
    var = np.array([1.0, 2.0], dtype=np.float64)
    m = np.array([0.0, 0.0], dtype=np.float64)
    v = np.array([0.0, 0.0], dtype=np.float64)
    beta1_power = np.array(0.7, dtype=np.float64)
    beta2_power = np.array(0.95, dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    beta1 = np.array(0.7, dtype=np.float64)
    beta2 = np.array(0.95, dtype=np.float64)
    epsilon = np.array(1e-08, dtype=np.float64)
    grad = np.array([0.2, 0.3], dtype=np.float64)
    use_locking = False
    use_nesterov = True
    name = "adam_5"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": tf.constant(beta1_power),
        "beta2_power": tf.constant(beta2_power),
        "lr": tf.constant(lr),
        "beta1": tf.constant(beta1),
        "beta2": tf.constant(beta2),
        "epsilon": tf.constant(epsilon),
        "grad": tf.constant(grad),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64, with negative gradient
    var = np.array([1, 2], dtype=np.int64)
    m = np.array([0, 0], dtype=np.int64)
    v = np.array([0, 0], dtype=np.int64)
    beta1_power = np.array(0.9, dtype=np.int64)
    beta2_power = np.array(0.999, dtype=np.int64)
    lr = np.array(0.001, dtype=np.int64)
    beta1 = np.array(0.9, dtype=np.int64)
    beta2 = np.array(0.999, dtype=np.int64)
    epsilon = np.array(1, dtype=np.int64)
    grad = np.array([-1, -2], dtype=np.int64)
    use_locking = False
    use_nesterov = False
    name = "adam_6"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": tf.constant(beta1_power),
        "beta2_power": tf.constant(beta2_power),
        "lr": tf.constant(lr),
        "beta1": tf.constant(beta1),
        "beta2": tf.constant(beta2),
        "epsilon": tf.constant(epsilon),
        "grad": tf.constant(grad),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8
    var = np.array([1, 2], dtype=np.uint8)
    m = np.array([0, 0], dtype=np.uint8)
    v = np.array([0, 0], dtype=np.uint8)
    beta1_power = np.array(0.9, dtype=np.uint8)
    beta2_power = np.array(0.999, dtype=np.uint8)
    lr = np.array(0.001, dtype=np.uint8)
    beta1 = np.array(0.9, dtype=np.uint8)
    beta2 = np.array(0.999, dtype=np.uint8)
    epsilon = np.array(1, dtype=np.uint8)
    grad = np.array([1, 2], dtype=np.uint8)
    use_locking = False
    use_nesterov = False
    name = "adam_7"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": tf.constant(beta1_power),
        "beta2_power": tf.constant(beta2_power),
        "lr": tf.constant(lr),
        "beta1": tf.constant(beta1),
        "beta2": tf.constant(beta2),
        "epsilon": tf.constant(epsilon),
        "grad": tf.constant(grad),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, larger values
    var = np.array([100.0, 200.0], dtype=np.float32)
    m = np.array([0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 0.0], dtype=np.float32)
    beta1_power = np.array(0.9, dtype=np.float32)
    beta2_power = np.array(0.999, dtype=np.float32)
    lr = np.array(0.001, dtype=np.float32)
    beta1 = np.array(0.9, dtype=np.float32)
    beta2 = np.array(0.999, dtype=np.float32)
    epsilon = np.array(1e-07, dtype=np.float32)
    grad = np.array([10.0, 20.0], dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "adam_8"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": tf.constant(beta1_power),
        "beta2_power": tf.constant(beta2_power),
        "lr": tf.constant(lr),
        "beta1": tf.constant(beta1),
        "beta2": tf.constant(beta2),
        "epsilon": tf.constant(epsilon),
        "grad": tf.constant(grad),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex64
    var = np.array([1.0+1j, 2.0+2j], dtype=np.complex64)
    m = np.array([0.0+0j, 0.0+0j], dtype=np.complex64)
    v = np.array([0.0+0j, 0.0+0j], dtype=np.complex64)
    beta1_power = np.array(0.9, dtype=np.float32).astype(np.complex64)
    beta2_power = np.array(0.999, dtype=np.float32).astype(np.complex64)
    lr = np.array(0.001, dtype=np.float32).astype(np.complex64)
    beta1 = np.array(0.9, dtype=np.float32).astype(np.complex64)
    beta2 = np.array(0.999, dtype=np.float32).astype(np.complex64)
    epsilon = np.array(1e-07, dtype=np.float32).astype(np.complex64)
    grad = np.array([0.1+0.1j, 0.2+0.2j], dtype=np.complex64)
    use_locking = False
    use_nesterov = False
    name = "adam_9"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": tf.constant(beta1_power),
        "beta2_power": tf.constant(beta2_power),
        "lr": tf.constant(lr),
        "beta1": tf.constant(beta1),
        "beta2": tf.constant(beta2),
        "epsilon": tf.constant(epsilon),
        "grad": tf.constant(grad),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half (float16)
    var = np.array([1.0, 2.0], dtype=np.float16)
    m = np.array([0.0, 0.0], dtype=np.float16)
    v = np.array([0.0, 0.0], dtype=np.float16)
    beta1_power = np.array(0.9, dtype=np.float16)
    beta2_power = np.array(0.999, dtype=np.float16)
    lr = np.array(0.001, dtype=np.float16)
    beta1 = np.array(0.9, dtype=np.float16)
    beta2 = np.array(0.999, dtype=np.float16)
    epsilon = np.array(1e-07, dtype=np.float16)
    grad = np.array([0.1, 0.2], dtype=np.float16)
    use_locking = False
    use_nesterov = False
    name = "adam_10"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": tf.constant(beta1_power),
        "beta2_power": tf.constant(beta2_power),
        "lr": tf.constant(lr),
        "beta1": tf.constant(beta1),
        "beta2": tf.constant(beta2),
        "epsilon": tf.constant(epsilon),
        "grad": tf.constant(grad),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyAdam"] = tf_raw_ops_ApplyAdam_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyAdam' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdam'.")

check_valid('tf.raw_ops.ApplyAdam', generated_inputs['tf.raw_ops.ApplyAdam'], lib="tf", suffix=0)
