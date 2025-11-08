
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_Lgamma_inputs():
    list_of_inputs = []

    x = np.array(5.0, dtype=np.float32)
    name = "lgamma_scalar_f32"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 0.5, 1.0, 4.5, -4.0, -5.6], dtype=np.float32)
    name = "lgamma_vector_examples_f32"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-0.5, -1.3, -2.7], dtype=np.float64)
    name = "lgamma_neg_nonints_f64"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0, -10.0], dtype=np.float64)
    name = "lgamma_neg_integers_f64"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.1, 1.2, 2.3], [3.4, 10.5, 20.0]], dtype=np.float16)
    name = "lgamma_matrix_f16"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([
        [[1e-7, 2.0, -0.1], [0.0, 3.14, -3.14]],
        [[1.5, 2.5, 3.5], [-0.25, -0.75, 0.75]]
    ], dtype=np.float32)
    name = "lgamma_3d_mixed_f32"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[0.25, 0.75, 1.25]]], [[[5.5, -7.2, 12.0]]]], dtype=np.float64)
    name = "lgamma_4d_f64"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    name = "lgamma_empty_f32"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10.0, 50.0, 100.0], dtype=np.float64)
    name = "lgamma_large_values_f64"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    x = (base + 0.5)[:, ::2]
    name = "lgamma_noncontiguous_f32"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, np.inf, -np.inf, 2.0, -2.5], dtype=np.float32)
    name = "lgamma_special_values_f32"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1e-6, 1e-6, 1e-12, -1e-12], dtype=np.float64)
    name = "lgamma_tiny_values_f64"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Lgamma"] = tf_raw_ops_Lgamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Lgamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Lgamma', generated_inputs['tf.raw_ops.Lgamma'], lib="tf", suffix=0)
