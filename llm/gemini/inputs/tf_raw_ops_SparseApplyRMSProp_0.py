
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.compat.v1.disable_eager_execution()
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_SparseApplyRMSProp_inputs():
    list_of_inputs = []

    def create_input_dict(dtype):
        if dtype == np.float32:
            var_init = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            ms_init = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            mom_init = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
            lr_val = np.array(0.01, dtype=np.float32)
            rho_val = np.array(0.9, dtype=np.float32)
            momentum_val = np.array(0.0, dtype=np.float32)
            epsilon_val = np.array(1e-7, dtype=np.float32)
            grad_val = np.array([0.5, 0.0, 0.5, 0.0], dtype=np.float32)
            indices_val = np.array([0, 2], dtype=np.int32)
        else:
            var_init = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
            ms_init = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
            mom_init = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
            lr_val = np.array(0.01, dtype=np.float64)
            rho_val = np.array(0.9, dtype=np.float64)
            momentum_val = np.array(0.0, dtype=np.float64)
            epsilon_val = np.array(1e-7, dtype=np.float64)
            grad_val = np.array([0.5, 0.0, 0.5, 0.0], dtype=np.float64)
            indices_val = np.array([0, 2], dtype=np.int64)

        var = tf.compat.v1.get_variable("var", shape=(4,), dtype=dtype, initializer=tf.constant_initializer(var_init))
        ms = tf.compat.v1.get_variable("ms", shape=(4,), dtype=dtype, initializer=tf.constant_initializer(ms_init))
        mom = tf.compat.v1.get_variable("mom", shape=(4,), dtype=dtype, initializer=tf.constant_initializer(mom_init))
        lr = tf.constant(lr_val)
        rho = tf.constant(rho_val)
        momentum = tf.constant(momentum_val)
        epsilon = tf.constant(epsilon_val)
        grad = tf.constant(grad_val)
        indices = tf.constant(indices_val)
        use_locking = False
        name = "sparse_apply_rmsprop"

        input_dict = {
            "var": var,
            "ms": ms,
            "mom": mom,
            "lr": lr,
            "rho": rho,
            "momentum": momentum,
            "epsilon": epsilon,
            "grad": grad,
            "indices": indices,
            "use_locking": use_locking,
            "name": name,
        }
        return input_dict

    input_dict_float32 = create_input_dict(tf.float32)
    list_of_inputs.append(input_dict_float32)

    input_dict_float64 = create_input_dict(tf.float64)
    list_of_inputs.append(input_dict_float64)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseApplyRMSProp"] = tf_raw_ops_SparseApplyRMSProp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyRMSProp'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseApplyRMSProp', generated_inputs['tf.raw_ops.SparseApplyRMSProp'], lib="tf", suffix=0)
