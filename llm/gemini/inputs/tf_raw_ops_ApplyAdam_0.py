
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_apply_adam_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    m = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    beta1_power = np.array(0.9, dtype=np.float32)
    beta2_power = np.array(0.999, dtype=np.float32)
    lr = np.array(0.001, dtype=np.float32)
    beta1 = np.array(0.9, dtype=np.float32)
    beta2 = np.array(0.999, dtype=np.float32)
    epsilon = np.array(1e-07, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "adam_1"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "m": tf.Variable(m, dtype=tf.float32),
        "v": tf.Variable(v, dtype=tf.float32),
        "beta1_power": tf.constant(beta1_power, dtype=tf.float32),
        "beta2_power": tf.constant(beta2_power, dtype=tf.float32),
        "lr": tf.constant(lr, dtype=tf.float32),
        "beta1": tf.constant(beta1, dtype=tf.float32),
        "beta2": tf.constant(beta2, dtype=tf.float32),
        "epsilon": tf.constant(epsilon, dtype=tf.float32),
        "grad": tf.constant(grad, dtype=tf.float32),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2 - int32
    var = np.array([1, 2, 3], dtype=np.int32)
    m = np.array([0, 0, 0], dtype=np.int32)
    v = np.array([0, 0, 0], dtype=np.int32)
    beta1_power = np.array(0.9, dtype=np.float32).astype(np.int32)
    beta2_power = np.array(0.999, dtype=np.float32).astype(np.int32)
    lr = np.array(0.001, dtype=np.float32).astype(np.int32)
    beta1 = np.array(0.9, dtype=np.float32).astype(np.int32)
    beta2 = np.array(0.999, dtype=np.float32).astype(np.int32)
    epsilon = np.array(1e-07, dtype=np.float32).astype(np.int32)
    grad = np.array([1, 2, 3], dtype=np.int32)
    use_locking = True
    use_nesterov = True
    name = "adam_2"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.int32),
        "m": tf.Variable(m, dtype=tf.int32),
        "v": tf.Variable(v, dtype=tf.int32),
        "beta1_power": tf.constant(beta1_power, dtype=tf.int32),
        "beta2_power": tf.constant(beta2_power, dtype=tf.int32),
        "lr": tf.constant(lr, dtype=tf.int32),
        "beta1": tf.constant(beta1, dtype=tf.int32),
        "beta2": tf.constant(beta2, dtype=tf.int32),
        "epsilon": tf.constant(epsilon, dtype=tf.int32),
        "grad": tf.constant(grad, dtype=tf.int32),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 - float64, different shapes
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    m = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    v = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    beta1_power = np.array(0.9, dtype=np.float64)
    beta2_power = np.array(0.999, dtype=np.float64)
    lr = np.array(0.001, dtype=np.float64)
    beta1 = np.array(0.9, dtype=np.float64)
    beta2 = np.array(0.999, dtype=np.float64)
    epsilon = np.array(1e-08, dtype=np.float64)
    grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    use_locking = False
    use_nesterov = True
    name = "adam_3"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float64),
        "m": tf.Variable(m, dtype=tf.float64),
        "v": tf.Variable(v, dtype=tf.float64),
        "beta1_power": tf.constant(beta1_power, dtype=tf.float64),
        "beta2_power": tf.constant(beta2_power, dtype=tf.float64),
        "lr": tf.constant(lr, dtype=tf.float64),
        "beta1": tf.constant(beta1, dtype=tf.float64),
        "beta2": tf.constant(beta2, dtype=tf.float64),
        "epsilon": tf.constant(epsilon, dtype=tf.float64),
        "grad": tf.constant(grad, dtype=tf.float64),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - float16
    var = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    m = np.array([0.0, 0.0, 0.0], dtype=np.float16)
    v = np.array([0.0, 0.0, 0.0], dtype=np.float16)
    beta1_power = np.array(0.9, dtype=np.float16)
    beta2_power = np.array(0.999, dtype=np.float16)
    lr = np.array(0.001, dtype=np.float16)
    beta1 = np.array(0.9, dtype=np.float16)
    beta2 = np.array(0.999, dtype=np.float16)
    epsilon = np.array(1e-07, dtype=np.float16)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    use_locking = False
    use_nesterov = False
    name = "adam_4"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float16),
        "m": tf.Variable(m, dtype=tf.float16),
        "v": tf.Variable(v, dtype=tf.float16),
        "beta1_power": tf.constant(beta1_power, dtype=tf.float16),
        "beta2_power": tf.constant(beta2_power, dtype=tf.float16),
        "lr": tf.constant(lr, dtype=tf.float16),
        "beta1": tf.constant(beta1, dtype=tf.float16),
        "beta2": tf.constant(beta2, dtype=tf.float16),
        "epsilon": tf.constant(epsilon, dtype=tf.float16),
        "grad": tf.constant(grad, dtype=tf.float16),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - int64
    var = np.array([1, 2, 3], dtype=np.int64)
    m = np.array([0, 0, 0], dtype=np.int64)
    v = np.array([0, 0, 0], dtype=np.int64)
    beta1_power = np.array(0.9, dtype=np.float64).astype(np.int64)
    beta2_power = np.array(0.999, dtype=np.float64).astype(np.int64)
    lr = np.array(0.001, dtype=np.float64).astype(np.int64)
    beta1 = np.array(0.9, dtype=np.float64).astype(np.int64)
    beta2 = np.array(0.999, dtype=np.float64).astype(np.int64)
    epsilon = np.array(1e-07, dtype=np.float64).astype(np.int64)
    grad = np.array([1, 2, 3], dtype=np.int64)
    use_locking = True
    use_nesterov = True
    name = "adam_5"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.int64),
        "m": tf.Variable(m, dtype=tf.int64),
        "v": tf.Variable(v, dtype=tf.int64),
        "beta1_power": tf.constant(beta1_power, dtype=tf.int64),
        "beta2_power": tf.constant(beta2_power, dtype=tf.int64),
        "lr": tf.constant(lr, dtype=tf.int64),
        "beta1": tf.constant(beta1, dtype=tf.int64),
        "beta2": tf.constant(beta2, dtype=tf.int64),
        "epsilon": tf.constant(epsilon, dtype=tf.int64),
        "grad": tf.constant(grad, dtype=tf.int64),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - uint8
    var = np.array([1, 2, 3], dtype=np.uint8)
    m = np.array([0, 0, 0], dtype=np.uint8)
    v = np.array([0, 0, 0], dtype=np.uint8)
    beta1_power = np.array(1, dtype=np.uint8)
    beta2_power = np.array(1, dtype=np.uint8)
    lr = np.array(1, dtype=np.uint8)
    beta1 = np.array(1, dtype=np.uint8)
    beta2 = np.array(1, dtype=np.uint8)
    epsilon = np.array(1, dtype=np.uint8)
    grad = np.array([1, 2, 3], dtype=np.uint8)
    use_locking = True
    use_nesterov = True
    name = "adam_6"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.uint8),
        "m": tf.Variable(m, dtype=tf.uint8),
        "v": tf.Variable(v, dtype=tf.uint8),
        "beta1_power": tf.constant(beta1_power, dtype=tf.uint8),
        "beta2_power": tf.constant(beta2_power, dtype=tf.uint8),
        "lr": tf.constant(lr, dtype=tf.uint8),
        "beta1": tf.constant(beta1, dtype=tf.uint8),
        "beta2": tf.constant(beta2, dtype=tf.uint8),
        "epsilon": tf.constant(epsilon, dtype=tf.uint8),
        "grad": tf.constant(grad, dtype=tf.uint8),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - int16, use negative values
    var = np.array([-1, 2, -3], dtype=np.int16)
    m = np.array([0, 0, 0], dtype=np.int16)
    v = np.array([0, 0, 0], dtype=np.int16)
    beta1_power = np.array(0.9, dtype=np.float32).astype(np.int16)
    beta2_power = np.array(0.999, dtype=np.float32).astype(np.int16)
    lr = np.array(0.001, dtype=np.float32).astype(np.int16)
    beta1 = np.array(0.9, dtype=np.float32).astype(np.int16)
    beta2 = np.array(0.999, dtype=np.float32).astype(np.int16)
    epsilon = np.array(1e-07, dtype=np.float32).astype(np.int16)
    grad = np.array([-1, 2, -3], dtype=np.int16)
    use_locking = False
    use_nesterov = True
    name = "adam_7"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.int16),
        "m": tf.Variable(m, dtype=tf.int16),
        "v": tf.Variable(v, dtype=tf.int16),
        "beta1_power": tf.constant(beta1_power, dtype=tf.int16),
        "beta2_power": tf.constant(beta2_power, dtype=tf.int16),
        "lr": tf.constant(lr, dtype=tf.int16),
        "beta1": tf.constant(beta1, dtype=tf.int16),
        "beta2": tf.constant(beta2, dtype=tf.int16),
        "epsilon": tf.constant(epsilon, dtype=tf.int16),
        "grad": tf.constant(grad, dtype=tf.int16),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8 - complex64
    var = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    m = np.array([0+0j, 0+0j, 0+0j], dtype=np.complex64)
    v = np.array([0+0j, 0+0j, 0+0j], dtype=np.complex64)
    beta1_power = np.array(0.9+0j, dtype=np.complex64)
    beta2_power = np.array(0.999+0j, dtype=np.complex64)
    lr = np.array(0.001+0j, dtype=np.complex64)
    beta1 = np.array(0.9+0j, dtype=np.complex64)
    beta2 = np.array(0.999+0j, dtype=np.complex64)
    epsilon = np.array(1e-07+0j, dtype=np.complex64)
    grad = np.array([0.1+0.1j, 0.2+0.2j, 0.3+0.3j], dtype=np.complex64)
    use_locking = False
    use_nesterov = False
    name = "adam_8"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.complex64),
        "m": tf.Variable(m, dtype=tf.complex64),
        "v": tf.Variable(v, dtype=tf.complex64),
        "beta1_power": tf.constant(beta1_power, dtype=tf.complex64),
        "beta2_power": tf.constant(beta2_power, dtype=tf.complex64),
        "lr": tf.constant(lr, dtype=tf.complex64),
        "beta1": tf.constant(beta1, dtype=tf.complex64),
        "beta2": tf.constant(beta2, dtype=tf.complex64),
        "epsilon": tf.constant(epsilon, dtype=tf.complex64),
        "grad": tf.constant(grad, dtype=tf.complex64),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - different shapes for grad
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    m = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    beta1_power = np.array(0.9, dtype=np.float32)
    beta2_power = np.array(0.999, dtype=np.float32)
    lr = np.array(0.001, dtype=np.float32)
    beta1 = np.array(0.9, dtype=np.float32)
    beta2 = np.array(0.999, dtype=np.float32)
    epsilon = np.array(1e-07, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "adam_9"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "m": tf.Variable(m, dtype=tf.float32),
        "v": tf.Variable(v, dtype=tf.float32),
        "beta1_power": tf.constant(beta1_power, dtype=tf.float32),
        "beta2_power": tf.constant(beta2_power, dtype=tf.float32),
        "lr": tf.constant(lr, dtype=tf.float32),
        "beta1": tf.constant(beta1, dtype=tf.float32),
        "beta2": tf.constant(beta2, dtype=tf.float32),
        "epsilon": tf.constant(epsilon, dtype=tf.float32),
        "grad": tf.constant(grad, dtype=tf.float32),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10 - complex128
    var = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    m = np.array([0+0j, 0+0j, 0+0j], dtype=np.complex128)
    v = np.array([0+0j, 0+0j, 0+0j], dtype=np.complex128)
    beta1_power = np.array(0.9+0j, dtype=np.complex128)
    beta2_power = np.array(0.999+0j, dtype=np.complex128)
    lr = np.array(0.001+0j, dtype=np.complex128)
    beta1 = np.array(0.9+0j, dtype=np.complex128)
    beta2 = np.array(0.999+0j, dtype=np.complex128)
    epsilon = np.array(1e-07+0j, dtype=np.complex128)
    grad = np.array([0.1+0.1j, 0.2+0.2j, 0.3+0.3j], dtype=np.complex128)
    use_locking = False
    use_nesterov = False
    name = "adam_10"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.complex128),
        "m": tf.Variable(m, dtype=tf.complex128),
        "v": tf.Variable(v, dtype=tf.complex128),
        "beta1_power": tf.constant(beta1_power, dtype=tf.complex128),
        "beta2_power": tf.constant(beta2_power, dtype=tf.complex128),
        "lr": tf.constant(lr, dtype=tf.complex128),
        "beta1": tf.constant(beta1, dtype=tf.complex128),
        "beta2": tf.constant(beta2, dtype=tf.complex128),
        "epsilon": tf.constant(epsilon, dtype=tf.complex128),
        "grad": tf.constant(grad, dtype=tf.complex128),
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyAdam"] = tf_raw_ops_apply_adam_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAdam' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdam'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ApplyAdam', generated_inputs['tf.raw_ops.ApplyAdam'], lib="tf", suffix=0)
