
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_digamma_inputs():
    list_of_inputs = []

    x = np.array(3.5, dtype=np.float32)
    name = "digamma_scalar_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([1.0, 2.0, 0.5, -0.5, -1.5], dtype=np.float32)
    name = "digamma_vector_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
    name = "digamma_matrix_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[-0.1, -0.2], [1.0, 10.0]]], dtype=np.float16)
    name = "digamma_3d_f16"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[[1.0], [2.0]], [[-2.5], [3.5]]]], dtype=np.float32)
    name = "digamma_4d_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([], dtype=np.float32)
    name = "digamma_empty_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([1e-6, 1e6, 12345.678], dtype=np.float64)
    name = "digamma_large_vals_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([-1.0001, -2.0001, -3.5, -0.9999, 0.0001], dtype=np.float32)
    name = "digamma_near_poles_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
    name = "digamma_nan_inf_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[[[0.5, 1.5]], [[2.5, 3.5]]], [[[4.5, 5.5]], [[6.5, 7.5]]]]], dtype=np.float64)
    name = "digamma_5d_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[[2.0]]]], dtype=np.float16)
    name = "digamma_singleton_f16"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Digamma"] = tf_raw_ops_digamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Digamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Digamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Digamma', generated_inputs['tf.raw_ops.Digamma'], lib="tf", suffix=0)
