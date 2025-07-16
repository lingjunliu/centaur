
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ApplyAdam_inputs():
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
    name = "adam_update_1"

    input_dict = {
        "var": var,
        "m": m,
        "v": v,
        "beta1_power": beta1_power,
        "beta2_power": beta2_power,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    var = np.array([1.0], dtype=np.float32)
    m = np.array([0.0], dtype=np.float32)
    v = np.array([0.0], dtype=np.float32)
    beta1_power = np.array(0.9, dtype=np.float32)
    beta2_power = np.array(0.999, dtype=np.float32)
    lr = np.array(0.001, dtype=np.float32)
    beta1 = np.array(0.9, dtype=np.float32)
    beta2 = np.array(0.999, dtype=np.float32)
    epsilon = np.array(1e-07, dtype=np.float32)
    grad = np.array([0.1], dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "adam_update_5"

    input_dict = {
        "var": var,
        "m": m,
        "v": v,
        "beta1_power": beta1_power,
        "beta2_power": beta2_power,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    m = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    v = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    beta1_power = np.array(0.9, dtype=np.float32)
    beta2_power = np.array(0.999, dtype=np.float32)
    lr = np.array(0.001, dtype=np.float32)
    beta1 = np.array(0.9, dtype=np.float32)
    beta2 = np.array(0.999, dtype=np.float32)
    epsilon = np.array(1e-08, dtype=np.float32)
    grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    use_locking = True
    use_nesterov = True
    name = "adam_update_6"

    input_dict = {
        "var": var,
        "m": m,
        "v": v,
        "beta1_power": beta1_power,
        "beta2_power": beta2_power,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    m = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    v = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    beta1_power = np.array(0.9, dtype=np.float64)
    beta2_power = np.array(0.999, dtype=np.float64)
    lr = np.array(0.001, dtype=np.float64)
    beta1 = np.array(0.9, dtype=np.float64)
    beta2 = np.array(0.999, dtype=np.float64)
    epsilon = np.array(1e-07, dtype=np.float64)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    use_locking = False
    use_nesterov = False
    name = "adam_update_7"

    input_dict = {
        "var": var,
        "m": m,
        "v": v,
        "beta1_power": beta1_power,
        "beta2_power": beta2_power,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name,
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
