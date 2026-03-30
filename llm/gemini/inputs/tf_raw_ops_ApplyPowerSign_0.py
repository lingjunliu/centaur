
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_apply_power_sign_inputs():
    list_of_inputs = []

    # Input 1
    var = tf.Variable(np.array([1.0, 2.0], dtype=np.float32))
    m = tf.Variable(np.array([0.0, 0.0], dtype=np.float32))
    lr = tf.constant(0.1, dtype=np.float32)
    logbase = tf.constant(1.0, dtype=np.float32)
    sign_decay = tf.constant(1.0, dtype=np.float32)
    beta = tf.constant(0.9, dtype=np.float32)
    grad = tf.constant(np.array([0.5, -0.5], dtype=np.float32))
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
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyPowerSign"] = tf_raw_ops_apply_power_sign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyPowerSign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyPowerSign'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ApplyPowerSign', generated_inputs['tf.raw_ops.ApplyPowerSign'], lib="tf", suffix=0)
