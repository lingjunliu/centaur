
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_add_sign_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    m = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    alpha = np.array(0.9, dtype=np.float32)
    sign_decay = np.array(0.8, dtype=np.float32)
    beta = np.array(0.99, dtype=np.float32)
    grad = np.array([0.5, -0.2, 0.1], dtype=np.float32)
    use_locking = False
    name = "add_sign_1"

    input_dict = {
        "var": var,
        "m": m,
        "lr": lr,
        "alpha": alpha,
        "sign_decay": sign_decay,
        "beta": beta,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    m = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    alpha = np.array(0.9, dtype=np.float64)
    sign_decay = np.array(0.8, dtype=np.float64)
    beta = np.array(0.99, dtype=np.float64)
    grad = np.array([[0.5, -0.2], [0.1, -0.3]], dtype=np.float64)
    use_locking = True
    name = "add_sign_2"

    input_dict = {
        "var": var,
        "m": m,
        "lr": lr,
        "alpha": alpha,
        "sign_decay": sign_decay,
        "beta": beta,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyAddSign"] = tf_raw_ops_apply_add_sign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyAddSign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAddSign'.")

check_valid('tf.raw_ops.ApplyAddSign', generated_inputs['tf.raw_ops.ApplyAddSign'], lib="tf", suffix=0)
