
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_adadelta_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    accum_update = np.array([0.01], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.5], dtype=np.float32)
    use_locking = False
    name = "adadelta_1"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyAdadelta"] = tf_raw_ops_apply_adadelta_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyAdadelta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdadelta'.")

check_valid('tf.raw_ops.ApplyAdadelta', generated_inputs['tf.raw_ops.ApplyAdadelta'], lib="tf", suffix=0)
