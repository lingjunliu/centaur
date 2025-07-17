
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_adam_inputs():
    list_of_inputs = []

    # Input 1: Basic test case
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
    name = "adam_update_1"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": beta1_power,
        "beta2_power": beta2_power,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different learning rate and gradient
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    m = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    beta1_power = np.array(0.9, dtype=np.float32)
    beta2_power = np.array(0.999, dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    beta1 = np.array(0.9, dtype=np.float32)
    beta2 = np.array(0.999, dtype=np.float32)
    epsilon = np.array(1e-07, dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "adam_update_2"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": beta1_power,
        "beta2_power": beta2_power,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With locking
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
    use_locking = True
    use_nesterov = False
    name = "adam_update_3"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": beta1_power,
        "beta2_power": beta2_power,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With Nesterov
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
    use_nesterov = True
    name = "adam_update_4"

    input_dict = {
        "var": tf.Variable(var),
        "m": tf.Variable(m),
        "v": tf.Variable(v),
        "beta1_power": beta1_power,
        "beta2_power": beta2_power,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "grad": grad,
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
    
    print("Valid")

if 'tf.raw_ops.ApplyAdam' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdam'.")

check_valid('tf.raw_ops.ApplyAdam', generated_inputs['tf.raw_ops.ApplyAdam'], lib="tf", suffix=0)
