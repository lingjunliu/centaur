
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_power_sign_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    m = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lr = np.array(0.1, dtype=np.float32)
    logbase = np.array(2.0, dtype=np.float32)
    sign_decay = np.array(0.9, dtype=np.float32)
    beta = np.array(0.9, dtype=np.float32)
    grad = np.array([0.5, -0.5, 0.2], dtype=np.float32)
    use_locking = False
    name = "power_sign_1"

    input_dict = {
        "var": var,
        "m": m,
        "lr": lr,
        "logbase": logbase,
        "sign_decay": sign_decay,
        "beta": beta,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    m = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    logbase = np.array(1.5, dtype=np.float64)
    sign_decay = np.array(0.8, dtype=np.float64)
    beta = np.array(0.8, dtype=np.float64)
    grad = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float64)
    use_locking = True
    name = "power_sign_2"

    input_dict = {
        "var": var,
        "m": m,
        "lr": lr,
        "logbase": logbase,
        "sign_decay": sign_decay,
        "beta": beta,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([1, 2, 3], dtype=np.int32)
    m = np.array([0, 0, 0], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    logbase = np.array(2, dtype=np.int32)
    sign_decay = np.array(1, dtype=np.int32)
    beta = np.array(1, dtype=np.int32)
    grad = np.array([1, -1, 0], dtype=np.int32)
    use_locking = False
    name = "power_sign_3"

    input_dict = {
        "var": var,
        "m": m,
        "lr": lr,
        "logbase": logbase,
        "sign_decay": sign_decay,
        "beta": beta,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = np.array([1.0], dtype=np.float32)
    m = np.array([0.0], dtype=np.float32)
    lr = np.array(0.5, dtype=np.float32)
    logbase = np.array(3.0, dtype=np.float32)
    sign_decay = np.array(0.7, dtype=np.float32)
    beta = np.array(0.6, dtype=np.float32)
    grad = np.array([0.2], dtype=np.float32)
    use_locking = True
    name = "power_sign_4"

    input_dict = {
        "var": var,
        "m": m,
        "lr": lr,
        "logbase": logbase,
        "sign_decay": sign_decay,
        "beta": beta,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    var = np.array([1+1j, 2-2j], dtype=np.complex64)
    m = np.array([0+0j, 0+0j], dtype=np.complex64)
    lr = np.array(0.1+0j, dtype=np.complex64)
    logbase = np.array(2.0+0j, dtype=np.complex64)
    sign_decay = np.array(0.9+0j, dtype=np.complex64)
    beta = np.array(0.9+0j, dtype=np.complex64)
    grad = np.array([0.5-0.5j, -0.5+0.2j], dtype=np.complex64)
    use_locking = False
    name = "power_sign_5"

    input_dict = {
        "var": var,
        "m": m,
        "lr": lr,
        "logbase": logbase,
        "sign_decay": sign_decay,
        "beta": beta,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = np.array([[1+1j, 2-2j], [3+3j, 4-4j]], dtype=np.complex128)
    m = np.array([[0+0j, 0+0j], [0+0j, 0+0j]], dtype=np.complex128)
    lr = np.array(0.01+0j, dtype=np.complex128)
    logbase = np.array(1.5+0j, dtype=np.complex128)
    sign_decay = np.array(0.8+0j, dtype=np.complex128)
    beta = np.array(0.8+0j, dtype=np.complex128)
    grad = np.array([[0.1-0.2j, -0.2+0.3j], [0.3-0.4j, -0.4+0.5j]], dtype=np.complex128)
    use_locking = True
    name = "power_sign_6"

    input_dict = {
        "var": var,
        "m": m,
        "lr": lr,
        "logbase": logbase,
        "sign_decay": sign_decay,
        "beta": beta,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    var = np.array([1, 2, 3], dtype=np.int64)
    m = np.array([0, 0, 0], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    logbase = np.array(2, dtype=np.int64)
    sign_decay = np.array(1, dtype=np.int64)
    beta = np.array(1, dtype=np.int64)
    grad = np.array([1, -1, 0], dtype=np.int64)
    use_locking = False
    name = "power_sign_7"

    input_dict = {
        "var": var,
        "m": m,
        "lr": lr,
        "logbase": logbase,
        "sign_decay": sign_decay,
        "beta": beta,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    m = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lr = np.array(-0.1, dtype=np.float32)
    logbase = np.array(2.0, dtype=np.float32)
    sign_decay = np.array(0.9, dtype=np.float32)
    beta = np.array(0.9, dtype=np.float32)
    grad = np.array([-0.5, 0.5, -0.2], dtype=np.float32)
    use_locking = False
    name = "power_sign_8"

    input_dict = {
        "var": var,
        "m": m,
        "lr": lr,
        "logbase": logbase,
        "sign_decay": sign_decay,
        "beta": beta,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    var = np.array([1, 2, 3], dtype=np.int16)
    m = np.array([0, 0, 0], dtype=np.int16)
    lr = np.array(1, dtype=np.int16)
    logbase = np.array(2, dtype=np.int16)
    sign_decay = np.array(1, dtype=np.int16)
    beta = np.array(1, dtype=np.int16)
    grad = np.array([1, -1, 0], dtype=np.int16)
    use_locking = False
    name = "power_sign_9"

    input_dict = {
        "var": var,
        "m": m,
        "lr": lr,
        "logbase": logbase,
        "sign_decay": sign_decay,
        "beta": beta,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyPowerSign"] = tf_raw_ops_apply_power_sign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyPowerSign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyPowerSign'.")

check_valid('tf.raw_ops.ApplyPowerSign', generated_inputs['tf.raw_ops.ApplyPowerSign'], lib="tf", suffix=0)
